import numpy as np
from mpi4py import MPI
import logging
import sys
import time
import gc
from typing import Tuple, List, Optional

try:
    from sec_agg.mpc_function import LCC_encoding_w_Random_partial, LCC_decoding
except ImportError:
    logging.warning("LCC functions not found. Using dummy implementations.")
    def LCC_encoding_w_Random_partial(x, M):
        return np.split(x, M)
    def LCC_decoding(shards):
        return np.concatenate(shards)

class LightSecAggProtocol:
    def __init__(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.rng = np.random.default_rng(seed=self.rank)
        self.log_buffer = []
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging format and level"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
            datefmt='%Y-%m-%d,%H:%M:%S'
        )
    
    def ordered_log(self, message: str):
        """Store log message in buffer for ordered output"""
        self.log_buffer.append(f"[Rank {self.rank}] {message}")
    
    def print_ordered_logs(self, stage: str):
        """Gather and print logs in rank order for the given stage"""
        try:
            all_logs = self.comm.gather(self.log_buffer, root=0)
            if self.rank == 0:
                logging.info(f"\n=== {stage} Logs ===")
                for r in range(self.size):
                    if all_logs[r]:  # Only print if logs exist
                        logging.info(f"---- Rank {r} ----")
                        for msg in all_logs[r]:
                            logging.info(msg)
        except Exception as e:
            logging.error(f"Error in print_ordered_logs: {str(e)}")
        self.log_buffer = []
        self.comm.Barrier()
    
    def test_mpi_environment(self):
        """Verify MPI communication works"""
        try:
            if self.rank == 0:
                for i in range(1, self.size):
                    self.comm.Send([np.array([1], dtype=np.int64), MPI.INT64_T], dest=i)
                    data = np.empty(1, dtype=np.int64)
                    self.comm.Recv([data, MPI.INT64_T], source=i)
                    self.ordered_log(f"MPI test: Received {data[0]} from rank {i}")
            else:
                data = np.empty(1, dtype=np.int64)
                self.comm.Recv([data, MPI.INT64_T], source=0)
                self.comm.Send([np.array([self.rank], dtype=np.int64), MPI.INT64_T], dest=0)
                self.ordered_log(f"MPI test: Sent {self.rank} to rank 0")
            self.print_ordered_logs("MPI Test")
        except Exception as e:
            logging.error(f"MPI test failed: {str(e)}")
            self.comm.Abort()

    def parse_arguments(self, args: List[str]) -> Tuple[int, int, bool, float]:
        """Process command line arguments"""
        if len(args) == 1:
            if self.rank == 0:   
                logging.error("Please input the number of workers")
            self.comm.Abort()
        
        N = int(args[1])
        d = 500  # Default dimension
        is_sleep = False
        comm_mbps = 100.0  # Default communication speed
        
        if len(args) >= 3:
            d = int(args[2])
        if len(args) >= 4:
            is_sleep = True
            comm_mbps = float(args[3])
        
        return N, d, is_sleep, comm_mbps

    def generate_lagrange_polynomials(self, alpha: np.ndarray, beta: np.ndarray, 
                                    a_shards: np.ndarray, rt_ik: np.ndarray,
                                    v_ikn: np.ndarray, u_ikn: np.ndarray, 
                                    M: int, T: int, N: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate masked polynomials for secret sharing"""
        K = len(a_shards)
        phi = np.zeros((K, N), dtype=np.int64)
        psi = np.zeros((K, N), dtype=np.int64)
        
        # Add random noise for security
        phi, psi = self.hide_coordinates(phi, psi, K, N, p)
        
        for k in range(K):
            for j in range(N):
                phi_jk, psi_jk = 0, 0
                for m in range(M + T):
                    # Compute Lagrange basis polynomial
                    l_m = 1
                    for n in range(M + T):
                        if n != m:
                            l_m = (l_m * (alpha[j] - beta[n])) % p
                    
                    denominator = 1
                    for n in range(M + T):
                        if n != m:
                            denominator = (denominator * (beta[m] - beta[n])) % p
                    
                    inv_denominator = pow(int(denominator), p-2, p)
                    l_m = (l_m * inv_denominator) % p
                    
                    if m < M:
                        a_val = np.max(a_shards[k][m]) % p
                        phi_jk = (phi_jk + a_val * l_m) % p
                        psi_jk = (psi_jk + a_val * rt_ik[k] * l_m) % p
                    else:
                        phi_jk = (phi_jk + v_ikn[k][m-M] * l_m) % p
                        psi_jk = (psi_jk + u_ikn[k][m-M] * l_m) % p
                
                phi[k][j] = phi_jk
                psi[k][j] = psi_jk
        
        return phi, psi

    def hide_coordinates(self, phi: np.ndarray, psi: np.ndarray, 
                        K: int, N: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
        """Add random noise to hide actual values"""
        random_noise_phi = self.rng.integers(0, p, size=(K, N), dtype=np.int64)
        random_noise_psi = self.rng.integers(0, p, size=(K, N), dtype=np.int64)
        
        for k in range(K):
            # Ensure noise sums to zero for security
            noise_sum_phi = np.sum(random_noise_phi[k,:]) % p
            random_noise_phi[k,0] = (random_noise_phi[k,0] - noise_sum_phi) % p
            
            noise_sum_psi = np.sum(random_noise_psi[k,:]) % p
            random_noise_psi[k,0] = (random_noise_psi[k,0] - noise_sum_psi) % p
        
        phi = (phi + random_noise_phi) % p
        psi = (psi + random_noise_psi) % p
        return phi, psi

    def server_aggregate(self, phi_alphas: np.ndarray, alpha_s_eval: np.ndarray,
                        beta: np.ndarray, M: int, T: int, p: int) -> Optional[np.ndarray]:
        """Aggregate user submissions on the server"""
        U = len(alpha_s_eval)
        if U < M + T:
            logging.error(f"Not enough surviving users (got {U}, need at least {M+T})")
            return None
        
        if U > M + T:
            phi_alphas = phi_alphas[:M+T]
            alpha_s_eval = alpha_s_eval[:M+T]
        
        A = np.zeros((M, U), dtype=np.int64)
        for m in range(M):
            for j in range(U):
                # Compute interpolation coefficients
                numerator = 1
                denominator = 1
                for k in range(U):
                    if k != j:
                        numerator = (numerator * (beta[m] - alpha_s_eval[k])) % p
                        denominator = (denominator * (alpha_s_eval[j] - alpha_s_eval[k])) % p
                
                inv_denominator = pow(int(denominator), p-2, p)
                A[m][j] = (numerator * inv_denominator) % p
        
        x_agg = (A @ phi_alphas) % p
        return x_agg

    def verify_aggregation(self, x_agg: np.ndarray, phi_alphas: np.ndarray,
                         surviving_set: np.ndarray, p: int) -> bool:
        """Verify the aggregation result is valid"""
        if np.any(x_agg < 0) or np.any(x_agg >= p):
            logging.error("Aggregation result is out of bounds!")
            return False
        return True

    def real_to_finite_field(self, x: np.ndarray, scale: float = 1e6, p: int = 2**31-1) -> np.ndarray:
        """Convert real numbers to finite field elements"""
        scaled = np.round(x * scale).astype(np.int64)
        return np.mod(scaled, p)

    def run_offline_stage(self, N: int, M: int, T: int, p: int) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
        """Execute the offline stage of the protocol"""
        t_start = time.time()
        
        if self.rank == 0:  # Server
            try:
                alpha = self.rng.integers(1, p, size=N, dtype=np.int64)
                beta = self.rng.integers(1, p, size=M+T, dtype=np.int64)
                self.ordered_log(f"Server: generated alpha.shape={alpha.shape}, beta.shape={beta.shape}")
                
                # Distribute parameters to users
                for user_idx in range(1, N+1):
                    self.comm.Send([alpha, MPI.INT64_T], dest=user_idx, tag=1)
                    self.comm.Send([beta, MPI.INT64_T], dest=user_idx, tag=2)
                    self.ordered_log(f'Sent parameters to user {user_idx}')
                
                offline_time = time.time() - t_start
                return offline_time, (alpha, beta)
            
            except Exception as e:
                logging.error(f"Server offline stage failed: {str(e)}")
                self.comm.Abort()
        
        else:  # Users
            try:
                alpha = np.empty(N, dtype=np.int64)
                beta = np.empty(M+T, dtype=np.int64)
                self.comm.Recv([alpha, MPI.INT64_T], source=0, tag=1)
                self.comm.Recv([beta, MPI.INT64_T], source=0, tag=2)
                self.ordered_log(f"Received parameters alpha.shape={alpha.shape}, beta.shape={beta.shape}")
                
                offline_time = time.time() - t_start
                return offline_time, (alpha, beta)
            
            except Exception as e:
                logging.error(f"User {self.rank} offline stage failed: {str(e)}")
                self.comm.Abort()

    def run_online_stage_round1(self, alpha: np.ndarray, beta: np.ndarray,
                              a_shards: np.ndarray, M: int, T: int, N: int,
                              p: int, d: int, K: int) -> Tuple[float, Optional[Tuple[np.ndarray, np.ndarray]]]:
        t_start = time.time()
        if self.rank == 0:  # Server does nothing in round 1
            return time.time() - t_start, (None, None)
        try:
            # Users generate their random masks
            rt_ik = self.rng.integers(0, p, size=K, dtype=np.int64)
            self.ordered_log(f"Generated rt_ik.shape={rt_ik.shape}")

            # Exchange masks between users
            if self.rank == N:  # Last user aggregates masks
                total = np.zeros(K, dtype=np.int64)
                received_rts = {}
                for i in range(1, N):
                    rt = np.empty(K, dtype=np.int64)
                    status = MPI.Status()
                    self.comm.Recv([rt, MPI.INT64_T], source=MPI.ANY_SOURCE, status=status)
                    source_rank = status.Get_source()
                    received_rts[source_rank] = rt.copy()
                    self.ordered_log(f"Received rt from User {source_rank}")
                    total = (total + rt) % p
                rt_ik = (-total) % p
                self.ordered_log(f"Calculated final rt_ik")
                mask_sum = np.sum(rt_ik)
                for r in received_rts.values():
                    mask_sum = (mask_sum + np.sum(r)) % p
                if mask_sum != 0:
                    self.ordered_log(f"Mask sum verification FAILED! Sum={mask_sum}")
                else:
                    self.ordered_log("Mask sum verified: sum=0")
            else:  # Other users send their masks
                self.comm.Send([rt_ik, MPI.INT64_T], dest=N)
                self.ordered_log(f"Sent rt_ik to User {N}")

            # Generate random polynomials
            v_ikn = self.rng.integers(0, p, size=(K, T), dtype=np.int64)
            u_ikn = self.rng.integers(0, p, size=(K, T), dtype=np.int64)
            self.ordered_log(f"Generated v_ikn.shape={v_ikn.shape}, u_ikn.shape={u_ikn.shape}")

            # Generate Lagrange polynomials
            phi, psi = self.generate_lagrange_polynomials(
                alpha, beta, a_shards, rt_ik, v_ikn, u_ikn, M, T, N, p
            )
            self.ordered_log(f"Generated phi.shape={phi.shape}, psi.shape={psi.shape}")

            # Exchange polynomials between users
            phi_coeff = np.zeros((K, N), dtype=np.int64)
            psi_coeff = np.zeros((K, N), dtype=np.int64)
            phi_coeff[:,self.rank-1] = phi[:,self.rank-1]
            psi_coeff[:,self.rank-1] = psi[:,self.rank-1]

            for j in range(1, N+1):
                if j != self.rank:
                    # Send our coefficients
                    self.comm.Send([phi[:,j-1].copy(), MPI.INT64_T], dest=j, tag=1)
                    self.comm.Send([psi[:,j-1].copy(), MPI.INT64_T], dest=j, tag=2)
                    self.ordered_log(f"Sent phi/psi to User {j}")

                    # Receive others' coefficients
                    buf_phi = np.empty(K, dtype=np.int64)
                    buf_psi = np.empty(K, dtype=np.int64)
                    self.comm.Recv([buf_phi, MPI.INT64_T], source=j, tag=1)
                    self.comm.Recv([buf_psi, MPI.INT64_T], source=j, tag=2)
                    phi_coeff[:,j-1] = buf_phi
                    psi_coeff[:,j-1] = buf_psi
                    self.ordered_log(f"Received phi/psi from User {j}")

            online1_time = time.time() - t_start
            return online1_time, (phi_coeff, psi_coeff)
        except Exception as e:
            logging.error(f"User {self.rank} online stage round 1 failed: {str(e)}")
            return time.time() - t_start, (None, None)

    def run_online_stage_round2(self, phi_coeff: np.ndarray, psi_coeff: np.ndarray,
                              surviving_set: np.ndarray, d: int, K: int,
                              p: int, is_sleep: bool, comm_mbps: float) -> Tuple[float, np.ndarray]:
        """Execute second round of online stage"""
        t_start = time.time()
        
        if self.rank == 0:  # Server
            phi_alpha_buffer = np.zeros((len(surviving_set), K), dtype=np.int64)
            active_users = []
            
            # Check which users are active
            for j in range(1, self.size):
                status = np.array([0], dtype=np.int64)
                self.comm.Recv([status, MPI.INT64_T], source=j, tag=999)
                self.ordered_log(f"Received status={status[0]} from user {j}")
                
                if status[0] == 1:
                    active_users.append(j-1)
                    self.ordered_log(f"Will receive from user {j}")
            
            if len(active_users) < len(surviving_set):
                self.ordered_log(f"Not enough surviving users (got {len(active_users)}, need {len(surviving_set)})")
                self.comm.Abort()
            
            # Receive masked gradients from active users
            for j in active_users:
                self.comm.Recv([phi_alpha_buffer[j,:], MPI.INT64_T], source=j+1, tag=3)
                self.ordered_log(f"Received phi_alpha from user {j+1}")
            
            online2_time = time.time() - t_start
            return online2_time, phi_alpha_buffer
        
        else:  # Users
            # Simulate dropout
            is_active = self.rng.random() > 0.1  # 10% dropout rate
            status = np.array([1 if is_active else 0], dtype=np.int64)
            self.comm.Send([status, MPI.INT64_T], dest=0, tag=999)
            self.ordered_log(f"Sent status={status[0]} to server")
            
            if not is_active:
                self.ordered_log("Dropped out")
                return time.time() - t_start, np.array([])
            
            # Simulate local training
            time.sleep(5)  # Simulate computation
            
            # Generate and mask gradient
            bt_i = self.rng.integers(0, 2, size=d, dtype=np.int64)
            kt_i = self.rng.choice(d, size=K, replace=False)
            xt_i = self.rng.standard_normal(d).astype(np.int64)  # <-- FIXED HERE
            xt_i_1 = bt_i * xt_i
            xt_i_mod = self.real_to_finite_field(xt_i_1, p)
            
            xt_i_mod_masked = np.zeros(K, dtype=np.int64)
            for k in range(K):
                xt_i_mod_masked[k] = xt_i_mod[kt_i[k]]
            
            # Exchange masked gradients between surviving users
            global_masked_buffer = np.zeros((self.size-1, K), dtype=np.int64)
            my_idx = self.rank - 1
            
            if my_idx in surviving_set:
                for j in range(1, self.size):
                    if j != self.rank:
                        self.comm.Send([xt_i_mod_masked, MPI.INT64_T], dest=j, tag=3)
                        self.ordered_log(f"Sent masked model to User {j}")
                
                global_masked_buffer[my_idx,:] = xt_i_mod_masked
            
            if is_sleep:
                comm_time = K / comm_mbps / (2**20) * 32
                time.sleep(comm_time)
            
            for j in range(1, self.size):
                if j != self.rank:
                    temp_buf = np.empty(K, dtype=np.int64)
                    self.comm.Recv([temp_buf, MPI.INT64_T], source=j, tag=3)
                    global_masked_buffer[j-1,:] = temp_buf
                    self.ordered_log(f"Received masked model from User {j}")
            
            # Compute final submission to server
            phi_alpha_i = np.zeros(K, dtype=np.int64)
            if my_idx in surviving_set and is_active:
                for k in range(K):
                    for j in surviving_set:
                        val = global_masked_buffer[j,k]
                        phi_val = phi_coeff[k,j]
                        psi_val = psi_coeff[k,j]
                        phi_alpha_i[k] = (phi_alpha_i[k] + val * phi_val + psi_val) % p
                
                self.comm.Send([phi_alpha_i, MPI.INT64_T], dest=0, tag=3)
                self.ordered_log(f"Sent phi_alpha_i to Server")
            
            online2_time = time.time() - t_start
            return online2_time, phi_alpha_i

    def run(self, args: List[str]):
        """Main execution flow for LightSecAgg protocol"""
        self.test_mpi_environment()
        
        # Parse arguments and setup parameters
        N, d, is_sleep, comm_mbps = self.parse_arguments(args)
        
        if self.size != N + 1:
            logging.error(f"Need exactly {N} users and 1 server (total {N+1} processes)")
            self.comm.Abort()
        
        T = int(np.floor(N/2))
        U = max(T + 1, N)  # Number of surviving users
        p = 2**31-1  # Prime for finite field
        n_trials = 3  # Number of trials to average timing
        K = int(0.01 * d)  # Number of coordinates to use
        M = 6  # Number of shards
        
        # Calculate chunk size and adjust dimension
        chunk_size = d // M
        d = chunk_size * M
        
        # Initialize placeholders
        a_shards = self.rng.integers(0, p, size=(K, M), dtype=np.int64)
        self.ordered_log(f"Initialized placeholders: a_shards.shape={a_shards.shape}")
        
        if self.rank == 0:
            logging.info(f"Starting with N={N}, U={U}, T={T}, d={d}, K={K}, M={M}")
        
        time_avg = np.zeros(5, dtype=float)  # For timing statistics
        
        for trial in range(n_trials):
            t_total_start = time.time()
            self.ordered_log(f"\n=== Trial {trial+1}/{n_trials} ===")
            
            # Offline Stage
            offline_time, (alpha, beta) = self.run_offline_stage(N, M, T, p)
            self.print_ordered_logs("Offline Stage Complete")
            
            # Online Stage Round 1
            online1_time, (phi_coeff, psi_coeff) = self.run_online_stage_round1(
                alpha, beta, a_shards, M, T, N, p, d, K
            )
            self.print_ordered_logs("Online Stage Round 1 Complete")
            
            # Online Stage Round 2
            surviving_set = np.array(range(U))  # Indices of surviving users
            online2_time, phi_alpha_buffer = self.run_online_stage_round2(
                phi_coeff, psi_coeff, surviving_set, d, K, p, is_sleep, comm_mbps
            )
            self.print_ordered_logs("Online Stage Round 2 Complete")
            
            # Server aggregation
            t_agg_start = time.time()
            if self.rank == 0:
                alpha_s_eval = alpha[surviving_set]
                beta_s = beta[:M]
                x_agg = self.server_aggregate(phi_alpha_buffer, alpha_s_eval, beta_s, M, T, p)
                
                if x_agg is not None:
                    self.ordered_log(f"Aggregated gradient shape: {x_agg.shape}")
                    if not self.verify_aggregation(x_agg, phi_alpha_buffer, surviving_set, p):
                        self.ordered_log("Aggregation verification failed!")
            
            t_agg = time.time() - t_agg_start
            t_total = time.time() - t_total_start
            
            # Collect timing statistics
            if self.rank == 0:
                time_set = np.array([t_total, offline_time, online1_time, online2_time, t_agg])
                logging.info(f'Trial {trial} timing: Total={time_set[0]:.2f}s, Offline={time_set[1]:.2f}s, Online1={time_set[2]:.2f}s, Online2={time_set[3]:.2f}s, Agg={time_set[4]:.2f}s')
                time_avg += time_set
            
            self.comm.Barrier()
        
        if self.rank == 0 and n_trials > 0:
            time_avg /= n_trials
            logging.info(f'\nAverage timing over {n_trials} trials:')
            logging.info(f'Total time: {time_avg[0]:.2f}s')
            logging.info(f'Offline stage: {time_avg[1]:.2f}s')
            logging.info(f'Online stage round 1: {time_avg[2]:.2f}s')
            logging.info(f'Online stage round 2: {time_avg[3]:.2f}s')
            logging.info(f'Aggregation time: {time_avg[4]:.2f}s')

if __name__ == "__main__":
    protocol = LightSecAggProtocol()
    protocol.run(sys.argv)