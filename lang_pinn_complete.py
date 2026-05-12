import torch
import torch.nn as nn
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import deepxde as dde
from typing import List, Dict, Tuple, Optional
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 全局参数配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# PINN架构参数
HIDDEN_LAYERS = 4
HIDDEN_UNITS = 128
ACTIVATION = nn.Tanh()  # 可根据PDE特性动态调整
LEARNING_RATE = 1e-3
EPOCHS = 10000

# 四大智能体配置
# PDE Agent配置
K = 5  # CoT轨迹采样数量
ALPHA = 0.6  # 符号等价性与语义一致性权重（论文默认0.6）
# PINN Agent配置
ARCHITECTURES = ["MLP", "CNN", "GNN", "Transformer"]  # 候选架构
WEIGHTS = [0.2, 0.3, 0.5]  # 周期性、几何复杂度、多尺度权重（论文优先多尺度）
# Code Agent配置
MODULES = ["model", "pde_loss", "data_preprocess", "training_loop", "validation"]
# Feedback Agent配置
TAU = 1e-4  # 损失收敛阈值
EPS = 1e-6  # 梯度健康下界
KAPPA = 1e2  # 梯度健康上界
ALPHA_ROBUST = 0.5  # 损失平滑度与梯度健康权重

# 工具函数（论文核心公式实现）
def symbolic_equivalence(e1: sp.Expr, e2: sp.Expr) -> float:
    """
    论文4.2节：符号等价性评分，基于抽象语法树（AST）匹配
    :param e1: 候选PDE表达式1
    :param e2: 候选PDE表达式2
    :return: 0~1之间的等价性评分
    """
    def get_ast_nodes(expr: sp.Expr) -> List[str]:
        """提取表达式的AST节点（算子、变量、常数）"""
        nodes = []
        if isinstance(expr, sp.Symbol):
            nodes.append(str(expr))
        elif isinstance(expr, sp.Number):
            nodes.append(str(expr))
        elif isinstance(expr, sp.Derivative):
            nodes.append(f"d{expr.args[0]}/d{expr.args[1][0]}")
            nodes.extend(get_ast_nodes(expr.args[0]))
        elif isinstance(expr, (sp.Add, sp.Mul, sp.Pow)):
            nodes.append(expr.func.__name__)
            for arg in expr.args:
                nodes.extend(get_ast_nodes(arg))
        return nodes
    
    nodes1 = get_ast_nodes(e1)
    nodes2 = get_ast_nodes(e2)
    matched = len(set(nodes1) & set(nodes2))
    max_nodes = max(len(nodes1), len(nodes2))
    return matched / max_nodes if max_nodes != 0 else 0.0

def semantic_consistency(desc1: str, desc2: str) -> float:
    """
    论文4.2节：语义一致性评分，基于TF-IDF余弦相似度
    :param desc1: PDE候选的语义描述1
    :param desc2: PDE候选的语义描述2
    :return: 0~1之间的语义相似度
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform([desc1, desc2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

def pde_feature_encoding(pde_expr: sp.Expr, domain: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """
    论文4.3节：PDE特征编码（周期性、几何复杂度、多尺度需求）
    :param pde_expr: 符号化PDE表达式
    :param domain: 求解域（如{"x": (0,1), "t": (0,1)}）
    :return: 3维特征向量 [f_per, f_geo, f_ms]
    """
    # 1. 周期性特征 f_per
    periodic_axes = 0
    for var, (low, high) in domain.items():
        # 简单判断：若边界条件为周期性（如u(0,t)=u(1,t)），此处简化为假设1D/2D周期性
        # 实际可根据PDE边界条件进一步优化
        if "periodic" in str(pde_expr):
            periodic_axes += 1
    f_per = periodic_axes / len(domain) if len(domain) > 0 else 0.0
    
    # 2. 几何复杂度 f_geo（简化实现，论文附录1有详细公式）
    # 0: 矩形域 0.3: 曲域 0.6: 多组件域 0.9: 高度不规则域
    if len(domain) == 1:
        f_geo = 0.0  # 1D为矩形域
    elif len(domain) == 2 and all(high - low == 1.0 for _, (low, high) in domain.items()):
        f_geo = 0.0  # 2D正方形矩形域
    else:
        f_geo = 0.3  # 曲域/不规则域
    
    # 3. 多尺度需求 f_ms（简化实现，论文附录1有详细公式）
    # 基于导数阶数、非线性项、Re/Pe数判断
    derivatives = list(pde_expr.atoms(sp.Derivative))
    max_order = max([d.args[1][1] for d in derivatives]) if derivatives else 1
    has_nonlinear = any(isinstance(arg, sp.Pow) for arg in pde_expr.atoms(sp.Pow) if arg.exp != 1)
    f_ms = (0.3 * (max_order >= 3) + 0.4 * has_nonlinear + 0.3 * (len(derivatives) >= 3))
    f_ms = 1 / (1 + np.exp(-f_ms))  # sigmoid归一化到[0,1]
    
    return np.array([f_per, f_geo, f_ms], dtype=np.float32)

def architecture_capability(arch: str) -> np.ndarray:
    """
    论文4.3节：架构能力编码，基于论文附录1的预设值
    :param arch: 候选架构（MLP/CNN/GNN/Transformer）
    :return: 3维能力向量 [a_per, a_geo, a_ms]
    """
    cap_map = {
        "MLP": [0.1, 0.2, 0.4],
        "CNN": [0.2, 0.4, 0.3],
        "GNN": [0.1, 0.8, 0.5],
        "Transformer": [0.2, 0.5, 0.7]
    }
    return np.array(cap_map[arch], dtype=np.float32)

# 四大智能体实现
class PDEAgent:
    def __init__(self, k: int = K, alpha: float = ALPHA):
        self.k = k  # CoT轨迹数量
        self.alpha = alpha  # 符号与语义权重
    
    def cot_rephrasing(self, task_desc: str) -> List[str]:
        """
        论文4.2节：CoT重写，将自然语言描述标准化
        :param task_desc: 原始自然语言任务描述
        :return: K个标准化的描述
        """
        # 简化实现：去除无关信息、标准化物理术语
        # 实际可替换为LLM调用（如GPT-4、Llama2）进行CoT重写
        clean_desc = re.sub(r"\(.*?\)", "", task_desc)  # 去除括号内无关信息
        clean_desc = re.sub(r"\s+", " ", clean_desc).strip()
        # 生成K个轻微变体（模拟不同CoT轨迹）
        rephrasings = []
        for i in range(self.k):
            if i == 0:
                rephrasings.append(clean_desc)
            else:
                # 简单替换同义词，模拟CoT多样性
                synonyms = {
                    "diffusion": "diffuse",
                    "temperature": "temp",
                    "boundary condition": "BC",
                    "initial condition": "IC"
                }
                temp_desc = clean_desc
                for syn, rep in synonyms.items():
                    if syn in temp_desc and np.random.random() > 0.5:
                        temp_desc = temp_desc.replace(syn, rep)
                rephrasings.append(temp_desc)
        return rephrasings
    
    def formulate_pde(self, normalized_desc: str) -> Tuple[sp.Expr, str]:
        """
        将标准化描述转化为符号化PDE表达式
        :param normalized_desc: 标准化自然语言描述
        :return: (PDE符号表达式, PDE语义摘要)
        """
        # 简化实现：基于关键词匹配常见PDE（论文中支持8类PDE，此处以热方程为例）
        # 实际可替换为LLM调用，生成符号化PDE
        x, t = sp.symbols("x t")
        u = sp.Function("u")(x, t)
        
        if "heat" in normalized_desc.lower() or "diffusion" in normalized_desc.lower():
            # 热方程：u_t = α * u_xx
            alpha = sp.Symbol("α")
            pde_expr = sp.diff(u, t) - alpha * sp.diff(u, x, 2)
            semantic_summary = "1D heat diffusion equation with u_t = α*u_xx"
        elif "burgers" in normalized_desc.lower():
            # 伯格斯方程：u_t + u*u_x = ν * u_xx
            nu = sp.Symbol("ν")
            pde_expr = sp.diff(u, t) + u * sp.diff(u, x) - nu * sp.diff(u, x, 2)
            semantic_summary = "1D Burgers equation with u_t + u*u_x = ν*u_xx"
        elif "poisson" in normalized_desc.lower():
            # 泊松方程：u_xx + u_yy = f(x,y)
            x, y = sp.symbols("x y")
            u = sp.Function("u")(x, y)
            f = sp.Function("f")(x, y)
            pde_expr = sp.diff(u, x, 2) + sp.diff(u, y, 2) - f
            semantic_summary = "2D Poisson equation with u_xx + u_yy = f(x,y)"
        else:
            # 默认热方程（可扩展）
            alpha = sp.Symbol("α")
            pde_expr = sp.diff(u, t) - alpha * sp.diff(u, x, 2)
            semantic_summary = "1D heat diffusion equation with u_t = α*u_xx"
        
        return pde_expr, semantic_summary
    
    def consensus_voting(self, candidates: List[Tuple[sp.Expr, str]]) -> Tuple[sp.Expr, str]:
        """
        论文4.2节：共识投票，选择最优PDE候选
        :param candidates: K个PDE候选（表达式+语义摘要）
        :return: 最优PDE（表达式+语义摘要）
        """
        n = len(candidates)
        scores = np.zeros(n)
        
        for i in range(n):
            e1, d1 = candidates[i]
            total_score = 0.0
            for j in range(n):
                if i == j:
                    continue
                e2, d2 = candidates[j]
                sym_score = symbolic_equivalence(e1, e2)
                sem_score = semantic_consistency(d1, d2)
                total_score += self.alpha * sym_score + (1 - self.alpha) * sem_score
            scores[i] = total_score / (n - 1) if n > 1 else 1.0
        
        best_idx = np.argmax(scores)
        return candidates[best_idx]
    
    def run(self, task_desc: str) -> Tuple[sp.Expr, str, Dict[str, Tuple[float, float]]]:
        """
        PDE Agent主流程
        :param task_desc: 原始自然语言任务描述
        :return: 最优PDE表达式、语义摘要、求解域
        """
        # 1. CoT重写
        normalized_descs = self.cot_rephrasing(task_desc)
        # 2. 生成PDE候选
        candidates = []
        for desc in normalized_descs:
            pde_expr, sem_summary = self.formulate_pde(desc)
            candidates.append((pde_expr, sem_summary))
        # 3. 共识投票
        best_pde, best_sem = self.consensus_voting(candidates)
        # 4. 确定求解域（简化实现，可根据自然语言进一步优化）
        if "1d" in best_sem.lower() or len(best_pde.atoms(sp.Symbol)) == 2:
            domain = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
        elif "2d" in best_sem.lower() or len(best_pde.atoms(sp.Symbol)) == 3:
            domain = {"x": (0.0, 1.0), "y": (0.0, 1.0)}
        else:
            domain = {"x": (0.0, 1.0)}
        
        print(f"PDE Agent: 生成最优PDE -> {best_pde}")
        return best_pde, best_sem, domain

class PINNAgent:
    def __init__(self, architectures: List[str] = ARCHITECTURES, weights: List[float] = WEIGHTS):
        self.architectures = architectures
        self.weights = np.array(weights, dtype=np.float32)  # [w_per, w_geo, w_ms]
        self.history_cache = {}  # 历史缓存：PDE语义摘要 -> 最优架构
    
    def history_reuse(self, pde_sem: str) -> Optional[str]:
        """
        论文4.3节：历史复用，若存在相似PDE则直接复用架构
        :param pde_sem: PDE语义摘要
        :return: 复用架构（None表示无复用）
        """
        # 简化实现：基于语义相似度判断
        for cached_sem, cached_arch in self.history_cache.items():
            if semantic_consistency(pde_sem, cached_sem) > 0.8:
                print(f"PINN Agent: 历史复用架构 -> {cached_arch}")
                return cached_arch
        return None
    
    def knowledge_guided_matching(self, pde_feature: np.ndarray) -> str:
        """
        论文4.3节：知识引导匹配，基于PDE特征与架构能力选择最优架构
        :param pde_feature: PDE特征向量 [f_per, f_geo, f_ms]
        :return: 最优架构
        """
        # 加权余弦相似度计算
        weighted_pde = self.weights * pde_feature
        similarities = []
        for arch in self.architectures:
            arch_cap = architecture_capability(arch)
            sim = np.dot(weighted_pde, arch_cap) / (np.linalg.norm(weighted_pde) * np.linalg.norm(arch_cap) + 1e-8)
            similarities.append(sim)
        
        best_arch = self.architectures[np.argmax(similarities)]
        print(f"PINN Agent: 知识引导选择架构 -> {best_arch}")
        return best_arch
    
    def run(self, pde_sem: str, pde_feature: np.ndarray) -> str:
        """
        PINN Agent主流程
        :param pde_sem: PDE语义摘要
        :param pde_feature: PDE特征向量
        :return: 最优PINN架构
        """
        # 1. 尝试历史复用
        reused_arch = self.history_reuse(pde_sem)
        if reused_arch is not None:
            return reused_arch
        # 2. 知识引导匹配
        best_arch = self.knowledge_guided_matching(pde_feature)
        # 3. 更新历史缓存
        self.history_cache[pde_sem] = best_arch
        return best_arch

class CodeAgent:
    def __init__(self, modules: List[str] = MODULES):
        self.modules = modules
    
    def generate_model(self, arch: str, input_dim: int, output_dim: int = 1) -> nn.Module:
        """
        生成PINN模型（模块化实现）
        :param arch: 选定的架构（MLP/CNN/GNN/Transformer）
        :param input_dim: 输入维度（如1D热方程：x,t -> 2维）
        :param output_dim: 输出维度（默认1，即u(x,t)）
        :return: PINN模型
        """
        if arch == "MLP":
            # MLP架构（论文默认基础架构）
            layers = []
            layers.append(nn.Linear(input_dim, HIDDEN_UNITS))
            layers.append(ACTIVATION)
            for _ in range(HIDDEN_LAYERS - 1):
                layers.append(nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS))
                layers.append(ACTIVATION)
            layers.append(nn.Linear(HIDDEN_UNITS, output_dim))
            model = nn.Sequential(*layers)
        elif arch == "CNN":
            # CNN架构（适配2D PDE）
            model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                ACTIVATION,
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                ACTIVATION,
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(64 * (input_dim//4) * (input_dim//4), HIDDEN_UNITS),
                ACTIVATION,
                nn.Linear(HIDDEN_UNITS, output_dim)
            )
        elif arch == "GNN":
            # GNN架构（适配不规则几何域，简化实现）
            from torch_geometric.nn import GCNConv, global_mean_pool
            class GNNModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.conv1 = GCNConv(input_dim, 64)
                    self.conv2 = GCNConv(64, 128)
                    self.fc = nn.Linear(128, output_dim)
                def forward(self, x, edge_index, batch):
                    x = self.conv1(x, edge_index)
                    x = ACTIVATION(x)
                    x = self.conv2(x, edge_index)
                    x = ACTIVATION(x)
                    x = global_mean_pool(x, batch)
                    x = self.fc(x)
                    return x
            model = GNNModel()
        elif arch == "Transformer":
            # Transformer架构（适配多尺度PDE）
            class TransformerModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embedding = nn.Linear(input_dim, HIDDEN_UNITS)
                    self.transformer = nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=HIDDEN_UNITS, nhead=4),
                        num_layers=2
                    )
                    self.fc = nn.Linear(HIDDEN_UNITS, output_dim)
                def forward(self, x):
                    x = self.embedding(x).unsqueeze(0)  # (seq_len, batch_size, d_model)
                    x = self.transformer(x)
                    x = x.squeeze(0)
                    return self.fc(x)
            model = TransformerModel()
        else:
            # 默认MLP
            layers = [nn.Linear(input_dim, HIDDEN_UNITS), ACTIVATION]
            for _ in range(HIDDEN_LAYERS - 1):
                layers.extend([nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS), ACTIVATION])
            layers.append(nn.Linear(HIDDEN_UNITS, output_dim))
            model = nn.Sequential(*layers)
        
        model.to(DEVICE)
        print(f"Code Agent: 生成{arch}模型，输入维度{input_dim}")
        return model
    
    def generate_pde_loss(self, pde_expr: sp.Expr, model: nn.Module, domain: Dict[str, Tuple[float, float]]) -> callable:
        """
        生成PDE损失函数（物理信息损失）
        :param pde_expr: 符号化PDE表达式
        :param model: 生成的PINN模型
        :param domain: 求解域
        :return: 损失函数
        """
        # 提取PDE中的变量（如x, t）
        symbols = list(pde_expr.atoms(sp.Symbol))
        input_vars = [sym for sym in symbols if not isinstance(sym, sp.Function) and str(sym) != "α" and str(sym) != "ν"]
        input_dim = len(input_vars)
        
        def pde_loss_func(x: torch.Tensor) -> torch.Tensor:
            """PDE残差损失"""
            x.requires_grad_(True)
            u = model(x)
            # 计算PDE残差（简化实现，以热方程为例）
            # 实际可通过sympy自动求导，适配任意PDE
            u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
            u_xx = torch.autograd.grad(
                torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1],
                x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0][:, 0:1]
            alpha = torch.tensor(0.1, device=DEVICE)  # 热扩散系数（可从PDE中提取）
            residual = u_t - alpha * u_xx
            return torch.mean(torch.square(residual))
        
        return pde_loss_func
    
    def generate_data(self, domain: Dict[str, Tuple[float, float]], num_samples: int = 1000) -> torch.Tensor:
        """生成训练数据（采样求解域内的点）"""
        input_dim = len(domain)
        data = []
        for var, (low, high) in domain.items():
            samples = np.random.uniform(low, high, num_samples)
            data.append(samples)
        data = np.stack(data, axis=1)
        return torch.tensor(data, dtype=torch.float32).to(DEVICE)
    
    def generate_training_loop(self, model: nn.Module, pde_loss: callable, train_data: torch.Tensor) -> callable:
        """生成训练循环"""
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        def train():
            model.train()
            losses = []
            for epoch in range(EPOCHS):
                optimizer.zero_grad()
                loss = pde_loss(train_data)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                if epoch % 1000 == 0:
                    print(f"Training Epoch {epoch:5d} | PDE Loss: {loss.item():.8f}")
                # 早停（满足收敛阈值）
                if loss.item() < TAU:
                    print(f"Training stopped early at epoch {epoch} (loss < {TAU})")
                    break
            return losses
        
        return train
    
    def run(self, arch: str, pde_expr: sp.Expr, domain: Dict[str, Tuple[float, float]]) -> Tuple[nn.Module, callable, torch.Tensor, callable]:
        """
        Code Agent主流程
        :param arch: 选定的PINN架构
        :param pde_expr: 符号化PDE表达式
        :param domain: 求解域
        :return: 模型、PDE损失、训练数据、训练函数
        """
        input_dim = len(domain)
        # 生成各模块
        model = self.generate_model(arch, input_dim)
        pde_loss = self.generate_pde_loss(pde_expr, model, domain)
        train_data = self.generate_data(domain)
        train_func = self.generate_training_loop(model, pde_loss, train_data)
        
        print("Code Agent: 模块化代码生成完成（模型、损失、数据、训练循环）")
        return model, pde_loss, train_data, train_func

class FeedbackAgent:
    def __init__(self, tau: float = TAU, eps: float = EPS, kappa: float = KAPPA, alpha_robust: float = ALPHA_ROBUST):
        self.tau = tau  # 收敛阈值
        self.eps = eps  # 梯度健康下界
        self.kappa = kappa  # 梯度健康上界
        self.alpha_robust = alpha_robust  # 损失平滑度与梯度健康权重
    
    def error_diagnosis(self, train_func: callable) -> Tuple[bool, Optional[str]]:
        """
        运行训练函数，诊断错误
        :param train_func: 训练函数
        :return: (是否成功, 错误信息/None)
        """
        try:
            losses = train_func()
            return True, losses
        except RuntimeError as e:
            error_msg = f"Runtime Error: {str(e)}"
            if "shape mismatch" in str(e):
                error_msg += " (可能是模型输入输出维度不匹配，建议检查PINN Agent的架构选择)"
            elif "gradient" in str(e).lower():
                error_msg += " (梯度异常，建议调整学习率或模型架构)"
            print(f"Feedback Agent: 诊断到错误 -> {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = f"Unknown Error: {str(e)}"
            print(f"Feedback Agent: 诊断到错误 -> {error_msg}")
            return False, error_msg
    
    def quality_evaluation(self, model: nn.Module, pde_loss: callable, train_data: torch.Tensor, losses: List[float]) -> Dict[str, float]:
        """
        论文4.5节：多维度质量评估（有效性、效率、鲁棒性）
        :param model: PINN模型
        :param pde_loss: PDE损失函数
        :param train_data: 训练数据
        :param losses: 训练损失序列
        :return: 质量评估分数（0~1）
        """
        # 1. 有效性（Effectiveness）：PDE残差误差（MSE）
        mse = pde_loss(train_data).item()
        effectiveness = 1 / (1 + np.log1p(mse))  # 归一化到[0,1]，误差越小分数越高
        
        # 2. 效率（Efficiency）：收敛速度
        conv_steps = next((i for i, loss in enumerate(losses) if loss < self.tau), len(losses))
        efficiency = 1 - (conv_steps / len(losses))  # 收敛越快分数越高
        
        # 3. 鲁棒性（Robustness）：损失平滑度 + 梯度健康
        # 3.1 损失平滑度
        if len(losses) < 2:
            loss_smoothness = 0.5
        else:
            delta_loss = np.diff(losses)
            std_delta = np.std(delta_loss)
            mean_loss = np.mean(losses)
            loss_smoothness = 1 - (std_delta / (mean_loss + 1e-8))
            loss_smoothness = np.clip(loss_smoothness, 0.0, 1.0)
        # 3.2 梯度健康
        model.train()
        train_data.requires_grad_(True)
        loss = pde_loss(train_data)
        loss.backward()
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = torch.norm(param.grad).item()
                grad_norms.append(grad_norm)
        if not grad_norms:
            gradient_health = 0.0
        else:
            avg_grad_norm = np.mean(grad_norms) / len(model.parameters())
            gradient_health = 1.0 if (self.eps <= avg_grad_norm <= self.kappa) else 0.0
        # 3.3 鲁棒性总分
        robustness = self.alpha_robust * loss_smoothness + (1 - self.alpha_robust) * gradient_health
        
        # 总体质量分数（加权和）
        quality_score = 0.4 * effectiveness + 0.3 * efficiency + 0.3 * robustness
        quality_score = np.clip(quality_score, 0.0, 1.0)
        
        evaluation = {
            "effectiveness": effectiveness,
            "efficiency": efficiency,
            "robustness": robustness,
            "quality_score": quality_score,
            "mse": mse,
            "conv_steps": conv_steps,
            "losses": losses
        }
        print(f"Feedback Agent: 质量评估 -> {evaluation}")
        return evaluation
    
    def iterative_refinement(self, success: bool, evaluation: Optional[Dict[str, float]], 
                            pde_agent: PDEAgent, pinn_agent: PINNAgent, code_agent: CodeAgent,
                            task_desc: str, pde_expr: sp.Expr, pde_sem: str, pde_feature: np.ndarray, domain: Dict[str, Tuple[float, float]]) -> Tuple[nn.Module, Dict[str, float]]:
        """
        迭代修正：若训练失败或质量不达标，调整各智能体输出
        :param success: 训练是否成功
        :param evaluation: 质量评估结果
        :param 各智能体实例
        :param 各环节输出
        :return: 最终模型、最终质量评估
        """
        max_refine = 3  # 论文默认最大迭代次数
        refine_count = 0
        current_pde = pde_expr
        current_pde_sem = pde_sem
        current_pde_feature = pde_feature
        current_domain = domain
        
        while not success and refine_count < max_refine:
            print(f"\nFeedback Agent: 开始第{refine_count+1}次迭代修正")
            # 1. 重新运行PDE Agent（调整CoT重写策略）
            current_pde, current_pde_sem, current_domain = pde_agent.run(task_desc)
            # 2. 重新计算PDE特征
            current_pde_feature = pde_feature_encoding(current_pde, current_domain)
            # 3. 重新运行PINN Agent（强制不复用历史）
            pinn_agent.history_cache.pop(current_pde_sem, None)
            current_arch = pinn_agent.run(current_pde_sem, current_pde_feature)
            # 4. 重新运行Code Agent（调整模型参数）
            model, pde_loss, train_data, train_func = code_agent.run(current_arch, current_pde, current_domain)
            # 5. 重新诊断
            success, result = self.error_diagnosis(train_func)
            if success:
                evaluation = self.quality_evaluation(model, pde_loss, train_data, result)
            refine_count += 1
        
        if not success:
            raise RuntimeError(f"经过{max_refine}次迭代修正，仍无法生成可训练的PINN模型，请检查任务描述或调整参数")
        
        return model, evaluation
    
    def run(self, model: nn.Module, pde_loss: callable, train_data: torch.Tensor, train_func: callable,
            pde_agent: PDEAgent, pinn_agent: PINNAgent, code_agent: CodeAgent,
            task_desc: str, pde_expr: sp.Expr, pde_sem: str, pde_feature: np.ndarray, domain: Dict[str, Tuple[float, float]]) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Feedback Agent主流程
        :return: 最终可训练模型、质量评估结果
        """
        print("Feedback Agent: 开始运行代码诊断与质量评估")
        # 1. 错误诊断
        success, result = self.error_diagnosis(train_func)
        # 2. 质量评估与迭代修正
        if success:
            evaluation = self.quality_evaluation(model, pde_loss, train_data, result)
            # 若质量分数过低，进行迭代修正
            if evaluation["quality_score"] < 0.6:
                print(f"Feedback Agent: 质量分数过低（{evaluation['quality_score']} < 0.6），开始迭代修正")
                model, evaluation = self.iterative_refinement(
                    success, evaluation, pde_agent, pinn_agent, code_agent,
                    task_desc, pde_expr, pde_sem, pde_feature, domain
                )
        else:
            # 训练失败，进行迭代修正
            model, evaluation = self.iterative_refinement(
                success, None, pde_agent, pinn_agent, code_agent,
                task_desc, pde_expr, pde_sem, pde_feature, domain
            )
        
        print("Feedback Agent: 流程完成，生成最终可训练PINN模型")
        return model, evaluation

# Lang-PINN 主流程（整合四大智能体）
class LangPINN:
    def __init__(self):
        # 初始化四大智能体
        self.pde_agent = PDEAgent()
        self.pinn_agent = PINNAgent()
        self.code_agent = CodeAgent()
        self.feedback_agent = FeedbackAgent()
    
    def run(self, task_desc: str) -> Tuple[nn.Module, Dict[str, float], sp.Expr, str]:
        """
        Lang-PINN主流程：输入自然语言，输出可训练PINN模型
        :param task_desc: 自然语言任务描述
        :return: 最终模型、质量评估、PDE表达式、PDE语义摘要
        """
        print("="*50)
        print("Lang-PINN 开始运行：从自然语言到PINN模型")
        print("="*50)
        
        # 1. PDE Agent：自然语言转PDE
        pde_expr, pde_sem, domain = self.pde_agent.run(task_desc)
        
        # 2. 计算PDE特征
        pde_feature = pde_feature_encoding(pde_expr, domain)
        print(f"PDE 特征向量 -> {pde_feature}")
        
        # 3. PINN Agent：选择架构
        arch = self.pinn_agent.run(pde_sem, pde_feature)
        
        # 4. Code Agent：生成模块化代码
        model, pde_loss, train_data, train_func = self.code_agent.run(arch, pde_expr, domain)
        
        # 5. Feedback Agent：诊断与迭代修正
        final_model, evaluation = self.feedback_agent.run(
            model, pde_loss, train_data, train_func,
            self.pde_agent, self.pinn_agent, self.code_agent,
            task_desc, pde_expr, pde_sem, pde_feature, domain
        )
        
        print("="*50)
        print("Lang-PINN 运行完成！")
        print(f"最终模型架构：{arch}")
        print(f"最终MSE：{evaluation['mse']:.8f}")
        print(f"质量评估分数：{evaluation['quality_score']:.4f}")
        print("="*50)
        
        return final_model, evaluation, pde_expr, pde_sem

# 测试运行（示例：1D热方程自然语言描述）
if __name__ == "__main__":
    # 示例：自然语言任务描述（1D热方程）
    task_description = """
    一个薄金属杆，长度为1单位，初始温度分布为x*(1-x)。
    杆的两端被固定在0温度，杆内的热量通过热传导扩散，
    请建立模型求解杆在不同时间点的温度分布，热扩散系数为0.1。
    """
    
    # 运行Lang-PINN
    lang_pinn = LangPINN()
    final_model, evaluation, pde_expr, pde_sem = lang_pinn.run(task_description)
    
    # 可视化训练损失
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation["losses"], label="Training Loss")
    plt.axhline(y=TAU, color="r", linestyle="--", label=f"Convergence Threshold ({TAU})")
    plt.xlabel("Epoch")
    plt.ylabel("PDE Loss (MSE)")
    plt.title("Lang-PINN Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 保存模型
    torch.save(final_model.state_dict(), "lang_pinn_model.pth")
    print("模型已保存为 lang_pinn_model.pth")
