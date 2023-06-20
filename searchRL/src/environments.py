import numpy as np
from sklearn.metrics import ndcg_score


class Environment:
    """Среда для воздействия на векторы запросов или документов, полученных языковой моделью.
    Каждому значению вектора соответствует одна ручка бандита и один коэффициент,
    на который умножается щначение вектора.
    Использование ручки меняет коэффициент (вверх или вниз) на значение value, после чего
    или пересчитывается вектор запроса и ищутся новые кандидаты, или пересчитывается расстояние до векторов кандидатов,
    в зависимости от выбранной стратегии.
    Если изменение коэффициента увеличило NDCG, ручка награждается.
    Args:
        index - построенный hnswlib-индекс
        model - обученная FastText модель
        scaler - обученный sklearn StandardScaler
        constant (np.array) - массив констант, на которые смещаются вектора после шкалирования.
            Необходимо для приведения всех чисел векторов к положительным значениям.
        items (np.array) - массив id товаров, соответствующих векторам индекса
        n_arms (int) - количество ручек. Должно соответствовать размеру векторов модели и индекса
        rewards (list[int]) - пара награда / штраф
        n_search (int) - количество кандидатов, возвращаемых индексом
        top_k (int) - топ товаров для расчета ndcg
        value (float) - величина, на которую изменяется коэффициент соотвтетсвующей ручки.
    """
    def __init__(
        self,
        index,
        model,
        scaler,
        constant: np.array,
        items: np.array,
        n_arms: int,
        rewards: list = [100, -1],
        n_search: int = 100,
        top_k: int = 20,
        value: float = 0.1
    ):
        assert n_arms == model.vector_size, "The number of arms is not compatible with vector size"
        self.index = index
        self.model = model
        self.scaler = scaler
        self.constant = constant
        self.items = items
        self.n_arms = n_arms
        self.rewards = rewards
        self.n_search = n_search
        self.top_k = top_k
        self.value = value
        self.arm_coefficients = np.ones(n_arms)
        self.arm_signs = np.ones(n_arms)
        self.total_score = 0

    def vectorize(self, tokens: list) -> np.array:
        """Векторизация токенов.
        Args:
            tokens (list) - токены запроса
        Returns:
            np.array - вектор запроса
        """
        return np.mean([self.model.wv[x] for x in tokens], axis=0)

    def get_masked_relevance(self, pred_ids: np.array, true_ids: np.array, true_relevance: np.array) -> np.array:
        """Пересчет true-релевантностей. Массив действительных релевантностей расширяется до размера массива
        кандидатов, релевантности непересекающихся документов зануляются.
        Args:
            pred_ids (np.array) - массив id найденных документов
            true_ids (np.array) - массив id действительных документов
            true_relevance (np.array) - массив релевантностей действительных документов
        Returns:
            np.array - расширенный массив действительных документов
        """
        mask = np.in1d(pred_ids, true_ids)
        true_relevance_masked = np.zeros((pred_ids.shape))
        true_relevance_masked[mask] = true_relevance[np.in1d(true_ids, pred_ids)]
        return true_relevance_masked

    def normalize_distances(self, distances: np.array) -> np.array:
        """Инвертация расстояний до кандидатов. Необходмо для использования расстояний в качестве релевантностей.
        Args:
            distances (np.array) - массив расстояний
        Returns:
            np.array - массив релевантностей
        """
        m = 1 if max(distances) == 0. else max(distances)
        return (1 - distances / m)

    def pull_arm_to_vector(
        self,
        arm_id: int,
        query: str,
        true_ids: np.array,
        true_relevance: np.array,
        states: np.array
    ) -> int:
        """Применение ручки к вектору запроса. Коэффициенты с положительной наградой применяются к
        вектору запроса, ищутся кандидаты и рассчитывается NDCG. После чего имитируется использование ручки -
        меняется соответствующий коэффициент на величину value и умножается на соответствующее значение вектора.
        Находятся новые кандидаты и считается новый NDCG. Если NDCG вырос - ручка награждается, нет - штрафуется.
        Args:
            arm_id (int) - id ручки
            query (str) - поисковый запрос
            true_ids (np.array) - массив действительных документов, соответствующих запросу
            true_relevance (np.array) - массив действительны релевантностей документов, соответствующих запросу
            states (np.array) - индексы коэффициентов с положительной наградой
        Returns:
            reward (int) - награда ручки
        """
        tokens = query.split()
        vector = self.vectorize(tokens).reshape(1, -1)
        vector = self.scaler.transform(vector) + self.constant
        vector[:, states] *= self.arm_coefficients[states]

        I, D = self.index.knn_query(vector, k=self.n_search)
        pred_relevance = self.normalize_distances(D[0])
        true_relevance_masked = self.get_masked_relevance(self.items[I[0]], true_ids, true_relevance)

        initial_score = ndcg_score(
            true_relevance_masked.reshape(1, -1), pred_relevance.reshape(1, -1), k=self.top_k
        )
        self.total_score += initial_score

        self.arm_coefficients[arm_id] = self.arm_coefficients[arm_id] + (self.arm_signs[arm_id] * self.value)
        vector[:, arm_id] *= self.arm_coefficients[arm_id]

        I, D = self.index.knn_query(vector, k=self.n_search)
        pred_relevance = self.normalize_distances(D[0])
        true_relevance_masked = self.get_masked_relevance(self.items[I[0]], true_ids, true_relevance)

        new_score = ndcg_score(
            true_relevance_masked.reshape(1, -1), pred_relevance.reshape(1, -1), k=self.top_k
        )

        reward = self.rewards[0] if new_score > initial_score else self.rewards[1]
        if reward <= 0:
            self.arm_signs[arm_id] *= -1

        return reward

    def pull_arm_to_candidates(
        self,
        arm_id: int,
        query: str,
        true_ids: np.array,
        true_relevance: np.array,
        states: np.array
    ) -> int:
        """Применение ручки к векторам кандидатов. Поиск кандидатов осуществляется только один ращ.
        Коэффициенты с положительной наградой применяются к векторам кандидтов и рассчитывается NDCG.
        После чего имитируется использование ручки - меняется соответствующий коэффициент на величину value
        и умножается на соответствующее значение векторов кандидатов. Пересчитывается рассточние и считается новый NDCG.
        Если NDCG вырос - ручка награждается, нет - штрафуется.
        Args:
            arm_id (int) - id ручки
            query (str) - поисковый запрос
            true_ids (np.array) - массив действительных документов, соответствующих запросу
            true_relevance (np.array) - массив действительны релевантностей документов, соответствующих запросу
            states (np.array) - индексы коэффициентов с положительной наградой
        Returns:
            reward (int) - награда ручки
        """
        tokens = query.split()
        vector = self.vectorize(tokens).reshape(1, -1)
        vector = self.scaler.transform(vector) + self.constant

        I, _ = self.index.knn_query(vector, k=self.n_search)
        candidates = np.asarray(self.index.get_items(I[0]))
        candidates[:, states] *= self.arm_coefficients[states]

        D = ((vector - candidates)**2).sum(axis=1)
        pred_relevance = self.normalize_distances(D)
        true_relevance_masked = self.get_masked_relevance(self.items[I[0]], true_ids, true_relevance)

        initial_score = ndcg_score(
            true_relevance_masked.reshape(1, -1), pred_relevance.reshape(1, -1), k=self.top_k
        )
        self.total_score += initial_score

        self.arm_coefficients[arm_id] = self.arm_coefficients[arm_id] + (self.arm_signs[arm_id] * self.value)
        candidates[:, arm_id] *= self.arm_coefficients[arm_id]
        D = ((vector - candidates)**2).sum(axis=1)
        pred_relevance = self.normalize_distances(D)

        new_score = ndcg_score(
            true_relevance_masked.reshape(1, -1), pred_relevance.reshape(1, -1), k=self.top_k
        )

        reward = self.rewards[0] if new_score > initial_score else self.rewards[1]
        if reward <= 0:
            self.arm_signs[arm_id] *= -1

        return reward

    def get_ndcg_to_vector(self, query: str, true_ids: np.array, true_relevance: np.array, states: np.array) -> float:
        """Расчет NDCG при воздействии на вектор запроса.
        Args:
            query (str) - поисковый запрос
            true_ids (np.array) - массив действительных документов, соответствующих запросу
            true_relevance (np.array) - массив действительны релевантностей документов, соответствующих запросу
            states (np.array) - индексы коэффициентов с положительной наградой
        Returns:
            float - NDCG
        """
        tokens = query.split()
        vector = self.vectorize(tokens).reshape(1, -1)
        vector = self.scaler.transform(vector) + self.constant
        vector[:, states] *= self.arm_coefficients[states]

        I, D = self.index.knn_query(vector, k=self.n_search)
        pred_relevance = self.normalize_distances(D[0])
        true_relevance_masked = self.get_masked_relevance(self.items[I[0]], true_ids, true_relevance)

        return ndcg_score(
            true_relevance_masked.reshape(1, -1), pred_relevance.reshape(1, -1), k=self.top_k
        )

    def get_ndcg_to_candidates(self, query: str, true_ids: np.array, true_relevance: np.array, states: np.array) -> float:
        """Расчет NDCG при воздействии на векторы кандидатов.
        Args:
            query (str) - поисковый запрос
            true_ids (np.array) - массив действительных документов, соответствующих запросу
            true_relevance (np.array) - массив действительны релевантностей документов, соответствующих запросу
            states (np.array) - индексы коэффициентов с положительной наградой
        Returns:
            float - NDCG
        """
        tokens = query.split()
        vector = self.vectorize(tokens).reshape(1, -1)
        vector = self.scaler.transform(vector) + self.constant
        I, _ = self.index.knn_query(vector, k=self.n_search)
        candidates = np.asarray(self.index.get_items(I[0]))
        candidates[:, states] *= self.arm_coefficients[states]

        D = ((vector - candidates)**2).sum(axis=1)
        pred_relevance = self.normalize_distances(D)
        true_relevance_masked = self.get_masked_relevance(self.items[I[0]], true_ids, true_relevance)

        return ndcg_score(
            true_relevance_masked.reshape(1, -1), pred_relevance.reshape(1, -1), k=self.top_k
        )


class Strategy:

    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        self.n_iters = 0
        self.arms_states = np.zeros(n_arms)    # награда за каждую ручку
        self.arms_actions = np.zeros(n_arms)   # сколько раз выбрали каждую ручку

    def reset(self):
        self.n_iters = 0
        self.arms_states = np.zeros(self.n_arms)
        self.arms_actions = np.zeros(self.n_arms)

    def update_reward(self, arm: int, reward: int):
        self.n_iters += 1
        self.arms_states[arm] += reward
        self.arms_actions[arm] += 1

    def choose_arm(self):
        raise NotImplementedError


class UCB1(Strategy):

    def choose_arm(self):
        if self.n_iters < self.n_arms:
            return self.n_iters
        else:
            return np.argmax(self.ucb())

    def ucb(self):
        ucb = self.arms_states / self.arms_actions
        ucb += np.sqrt(2 * np.log(self.n_iters) / self.arms_actions)
        return ucb


class Thompson(Strategy):

    def __init__(self, n_arms: int):
        super().__init__(n_arms)
        self.alphas = np.ones(self.n_arms)
        self.betas = np.ones(self.n_arms)

    def choose_arm(self):
        arm = np.argmax([np.random.beta(self.alphas[i], self.betas[i]) for i in range(self.n_arms)])
        return arm

    def update_reward(self, arm: int, reward: int):
        super().update_reward(arm, reward)
        self.alphas[arm] += reward
        self.betas[arm] += 1 - reward


class Agent:

    def __init__(self, env: Environment, strategy: Strategy, env_type: 0):
        self.env = env
        self.strategy = strategy
        self.env_type = env_type

    def action(self, query, true_ids, true_relevance):
        states = np.where(self.strategy.arms_states > 0)[0]
        arm = self.strategy.choose_arm()
        if self.env_type == 0:
            reward = self.env.pull_arm_to_vector(arm, query, true_ids, true_relevance, states)
        else:
            reward = self.env.pull_arm_to_candidates(arm, query, true_ids, true_relevance, states)
        self.strategy.update_reward(arm, reward)
