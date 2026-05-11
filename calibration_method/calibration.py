from scipy.optimize import minimize
import numpy as np


class CALIBRATION:

    # ------------------------------------------
    # 1. Function that computes B given its params
    # ------------------------------------------
    @staticmethod
    def compute_F(params, B):

        o1, o2, o3, s1, s2, s3, u1, u2, u3 = params
        S = np.array([[s1, 0, 0], [0, s2, 0], [0, 0, s3]])
        P = np.array(
            [
                [1, 0, 0],
                [-np.sin(u1), np.cos(u1), 0],
                [
                    np.sin(u2),
                    np.sin(u3),
                    np.sqrt(1 - np.sin(u2) ** 2 - np.sin(u3) ** 2),
                ],
            ]
        )
        O = np.array([o1, o2, o3])
        F_vec = S @ P @ B + O[:, None]

        return F_vec

    @staticmethod
    def compute_B(params, F):

        o1, o2, o3, s1, s2, s3, u1, u2, u3 = params

        S_inv = np.array([[s1**-1, 0, 0], [0, s2**-1, 0], [0, 0, s3**-1]])

        w = np.sqrt(1 - np.sin(u2) ** 2 - np.sin(u3) ** 2)
        if np.isnan(w):
            w = (2 - np.sin(u2) ** 2 - np.sin(u3) ** 2) / 2

        P_inv = np.array(
            [
                [1, 0, 0],  # [1,0,0],
                [np.sin(u1) / np.cos(u1), np.cos(u1) ** -1, 0],
                [
                    -(np.sin(u1) * np.sin(u3) + np.cos(u1) * np.sin(u2))
                    / (np.cos(u1) * w),
                    -(np.sin(u3)) / (np.cos(u1) * w),
                    1 / (w),
                ],
            ]
        )
        O = np.array([o1, o2, o3])

        B = P_inv @ S_inv @ (F - O[:, None])

        return B

    # ------------------------------------------
    # 2. Loss function with arguments
    # ------------------------------------------

    cost_log = []

    def Cost_Function(self, params, B_true, F, std_B):
        B = CALIBRATION.compute_B(params, F)
        B_norm = np.linalg.norm(B, axis=0)
        beta_scale = 0.1
        penalty_scale = beta_scale * np.sum(np.exp(np.abs((params[3:6])) - 2))
        r = np.sum((B_norm - B_true) ** 2) + penalty_scale
        # --------------------------------- cost log --------------------------------- #
        self.cost_log.append(np.sum((B_norm - B_true) ** 2))
        return r

    def Cost_Function_Fast(self, params, B_true, F, std_B):

        B = CALIBRATION.compute_B(params, F)
        B_norm = np.linalg.norm(B, axis=0)
        B_mean_diff_sum = np.exp(np.mean(B_norm - B_true)) - 1
        betta_offset = 1
        penalty_offset = betta_offset * (B_mean_diff_sum)
        beta_scale = 1
        penalty_scale = beta_scale * np.sum(np.exp(np.abs((params[3:6])) - 2))
        beta_std = 10
        penalty_std = beta_std * np.mean((B_norm - np.mean(B_norm)) ** 2)

        r = penalty_scale + penalty_offset + penalty_std

        # R = np.mean(r)
        # print("penalty ratios",penalty/R,penalty_scale/R,penalty_std/R)

        # --------------------------------- cost log --------------------------------- #
        self.cost_log.append(np.sum((B_norm - B_true) ** 2))

        return r

    # ------------------------------------------
    # 3. Optimize
    # ------------------------------------------

    def calibrate(self, x_axis, y_axis, z_axis, true_value, n_tryles=20, skip=False):
        # Nominal starting parameters
        m = np.array(
            [
                0.1,
                0.09,
                1.4,  # offsets (eu)
                1.0012681,
                0.9970246,
                0.9956139,  # scales (eu/nT)
                np.deg2rad(315.25 / 3600.0),
                np.deg2rad(65.40 / 3600.0),
                np.deg2rad(-44.17 / 3600.0),  # non-orthogonality angles (rad)
            ]
        )

        F = np.array([x_axis, y_axis, z_axis])
        B_true = true_value

        # Logical ranges for random initialization
        offsets_range = [-1, 1]  # example: offsets between 0 and 1
        scales_range = [0.95, 1.05]  # example: scales around 1
        angles_range = [-np.pi / 180, np.pi / 180]  # small angles in radians (~±1°)

        best_loss = np.inf
        best_params = None
        best_cost_log = []

        for _ in range(n_tryles):

            # Generate a random initial guess within ranges
            init = np.array(
                [
                    np.random.uniform(*offsets_range, size=3),
                    np.random.uniform(*scales_range, size=3),
                    np.random.uniform(*angles_range, size=3),
                ]
            ).ravel()

            # Run least_squares
            if skip:
                res = minimize(
                    fun=self.Cost_Function,
                    x0=init,
                    jac="2-point",
                    args=(B_true, F, 2),
                    # options = {'maxiter' : 10}
                )

            else:

                res = minimize(
                    fun=self.Cost_Function_Fast,
                    x0=init,
                    jac="2-point",
                    args=(B_true, F, 2),
                    # options = {'maxiter' : 10}
                )
                fast_iteration = len(self.cost_log)
                fast_x = res.x
                fast_last_cost = self.cost_log[-1]
                res = minimize(
                    fun=self.Cost_Function,
                    x0=res.x,
                    jac="2-point",
                    args=(B_true, F, 2),
                    # options = {'maxiter' : 10}
                )
            # Keep the best result
            if self.cost_log[-1] < best_loss:
                best_loss = self.cost_log[-1]
                best_params = res.x
                best_cost_log = self.cost_log.copy()

            if skip:
                print(
                    f"""
    # ---------------------------------------------------------------------------- #
    # --------------------------------- TRAIL LOG -------------------------------- #
    # ---------------------------------------------------------------------------- #
    number of refine iterations: {len(self.cost_log)}
    last cost: {self.cost_log[-1]}
    reffined parameters: {res.x}
                """
                )
            else:
                print(
                    f"""
    # ---------------------------------------------------------------------------- #
    # --------------------------------- TRAIL LOG -------------------------------- #
    # ---------------------------------------------------------------------------- #
    initial parameters: {init}
    number of Fast iterations: {fast_iteration}
    last Fast cost: {fast_last_cost}
    Fast parameters: {fast_x}
    # ---------------------------------------------------------------------------- #
    number of refine iterations: {len(self.cost_log) - fast_iteration}
    last cost: {self.cost_log[-1]}
    reffined parameters: {res.x}
                """
                )

            self.cost_log = []

        print("Best parameters =", best_params)
        print(best_cost_log[-1], best_cost_log[-1] - best_cost_log[-2])
        return best_params, best_cost_log
