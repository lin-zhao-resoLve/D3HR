import torch
import os
import sys
sys.path.append('/home/zhao.lin1/D3HR')
from generation.dit_inversion_save_statistic import sample
from diffusers import DiTPipeline, DDIMScheduler
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm import tqdm
import ipdb
from PIL import Image

def save_average_latents(input_dir, save_dir):
    latents_by_class = {}

    # Iterate over each class folder in input_dir
    for class_name in sorted(os.listdir(input_dir)):
        class_dir = os.path.join(input_dir, class_name)
        
        if os.path.isdir(class_dir):
            latents = []
            for file_name in sorted(os.listdir(class_dir)):
                if file_name.endswith('.pt'):
                    file_path = os.path.join(class_dir, file_name)
                    latent = torch.load(file_path)  # Load the latent .pth file
                    latents.append(latent)
            
            # Store the latent vectors for this class
            latents_tensor = torch.stack(latents)
            average_latent = torch.mean(latents_tensor, dim=0)

            # latents_by_class[class_name] = latents

            save_file_path = os.path.join(save_dir, f'{class_name}_average_latent.pth')
            torch.save(average_latent, save_file_path)
            print(f"Saved average latent for class '{class_name}' to {save_file_path}")
    return 

def load_mapping(mapping_file):
    new_mapping = {}
    with open(mapping_file, 'r') as file:
        data = json.load(file)
        if "tiny" in mapping_file:
            for index, line in enumerate(file):
                #Extract the wnid starting with 'n' from each line and subtract 1 from the line number.
                key = line.split()[0]
                new_mapping[key] = index 
        else:
            new_mapping = {item["wnid"]: item["index"] for item in data.values()}
    return new_mapping


def process_class(class_dir, folder_path):
    data_list = []
    class_path = os.path.join(folder_path, class_dir)
    for file_name in os.listdir(class_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(class_path, file_name)
            tensor = torch.load(file_path, map_location=torch.device('cpu'))
            data_list.append(torch.flatten(tensor, start_dim=1))

    if data_list: 
        data = torch.stack(data_list, dim=0)
        mean = data.mean(dim=0)
        variance = data.var(dim=0)
        return class_dir, {"mean": mean, "variance": variance}
    return class_dir, None  

def process_p_sample(class_dir, folder_path):
    data_list = []
    class_path = os.path.join(folder_path, class_dir)
    for file_name in os.listdir(class_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(class_path, file_name)
            tensor = torch.load(file_path, map_location=torch.device('cpu'))
            data_list.append(tensor.flatten())

    data = torch.vstack(data_list)
    return data


def kl_divergence(selected_points, mean, std, device):
    """
    Compute the KL divergence between the candidate distribution and the target Gaussian distribution.
    KL(P || Q) = 0.5 * [tr(Sigma_Q^-1 Sigma_P) + (mu_Q - mu_P)^T Sigma_Q^-1 (mu_Q - mu_P) - k + log(det(Sigma_Q) / det(Sigma_P))]
    Here, the target distribution Q is N(mean,cov), 
    and the sampling distribution P is estimated from the selected_points.
    """
    # k = mean.size(0)  # Feature dimension: 4090
    selected_mean = selected_points.mean(dim=0)
    selected_var = selected_points.var(dim=0)
    selected_std = torch.sqrt(selected_var)

    # Compute KL divergnece
    diff = mean - selected_mean
    log_sigma_ratio = torch.log(selected_std / std)
    variance_ratio = (std**2 + diff**2) / (2 * selected_std**2)
    kl = torch.sum(log_sigma_ratio + variance_ratio - 0.5)

    return kl.item()  # Return a scalar value

def kl_divergence_independent_batch(mean, std, samples, device):
    mean = mean.to(device)
    std = std.to(device)
    samples = samples.to(device)

    # Compute KL divergence
    diff = samples - mean
    term1 = torch.sum(diff**2, dim=1) / (2 * std**2)  
    kl_divs = term1 + torch.log(std) - 0.5            # log(std/std)=0
    return kl_divs

def sinkhorn(A, B, epsilon=0.1, max_iter=1000, tol=1e-9):
    """
    Estimate the Wasserstein distance using the Sinkhorn algorithm, which supports distributions with different numbers of samples.
    A, B: The two input distributions
    epsilon: Sinkhorn regularization parameter
    max_iter: Maximum number of iterations
    tol: Convergence tolerance
    """

    # The amount of samples
    n_a, n_b = A.size(0), B.size(0)
    
    # Define weights and ensure normalization.
    weight_a = torch.ones(n_a, device=A.device) / n_a
    weight_b = torch.ones(n_b, device=B.device) / n_b

    # Compute the distance matrix
    C = torch.cdist(A, B, p=2) ** 2  # Squared Euclidean distance

    # Initialize dual variables
    u = torch.zeros(n_a, device=A.device)
    v = torch.zeros(n_b, device=B.device)

    K = torch.exp(-C / epsilon)  # Regularized distance matrix

    for _ in range(max_iter):
        # Update u and c, consider weights simultaneously
        u_new = epsilon * torch.log(weight_a) - epsilon * torch.logsumexp(-K / epsilon + v.view(1, -1), dim=1)
        v_new = epsilon * torch.log(weight_b) - epsilon * torch.logsumexp(-K / epsilon + u_new.view(-1, 1), dim=0)

        # Check convergence
        if torch.max(torch.abs(u_new - u)) < tol and torch.max(torch.abs(v_new - v)) < tol:
            break

        u, v = u_new, v_new

    transport_cost = torch.sum(K * C)  
    wasserstein_distance = transport_cost + epsilon * (torch.sum(u * weight_a) + torch.sum(v * weight_b))

    return wasserstein_distance

def skewness(tensor):
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)
    n = tensor.size(0)
    skew = torch.sum(((tensor - mean) / std) ** 3, dim=0) * (n / ((n - 1) * (n - 2)))
    return skew

def kurtosis(tensor):
    mean = torch.mean(tensor, dim=0)
    std = torch.std(tensor, dim=0)
    n = tensor.size(0)
    kurt = torch.sum(((tensor - mean) / std) ** 4, dim=0) * (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) - (3 * (n - 1) ** 2) / ((n - 2) * (n - 3))
    return kurt


def skewness_batch(tensor):
    mean = torch.mean(tensor, dim=1, keepdim=True)  # shape: [20000, 1, 4096]
    std = torch.std(tensor, dim=1, keepdim=True)    # shape: [20000, 1, 4096]
    
    n = tensor.size(2)  # feature dimension: 4096
    skew = torch.sum(((tensor - mean) / std) ** 3, dim=2) * (n / ((n - 1) * (n - 2)))  # shape: [20000, 1]
    
    return skew


def evaluate_distribution(samples, mean, std):
    sample_mean = torch.mean(samples, dim=0)
    mean_diff = torch.norm(sample_mean - mean)

    sample_std = torch.std(samples, dim=0)
    std_diff = torch.norm(sample_std - std)

    sample_skew = skewness_batch(samples)

    skew_diff = torch.norm(torch.tensor(sample_skew) - 0)  # Sample Skewness close to 0
    # kurt_diff = torch.norm(torch.tensor(sample_kurt) - 3)  # Sample Kurtosis close to 3

    # Comprehensive evaluation: each component can be weighted as needed
    score = mean_diff + std_diff + 10*skew_diff
    return score

def select_algorithm(n_trials, n_samples, mean, std, device):
    best_score = float('inf')
    best_sample = None
    for _ in range(n_trials):
        samples = torch.normal(mean.expand(n_samples, -1), std.expand(n_samples, -1)).to(device)
        
        score = evaluate_distribution(samples, mean, std)
        print(score)

        # Choose sample with best score
        if score < best_score:
            best_score = score
            best_sample = samples

    return best_sample


def evaluate_distribution_batch(samples, mean, std):
    sample_mean = torch.mean(samples, dim=1) 
    mean_diff = torch.norm(sample_mean - mean, dim=1) 

    sample_std = torch.std(samples, dim=1)  
    std_diff = torch.norm(sample_std - std, dim=1) 

    # Compute samples Skewness
    sample_skew = skewness_batch(samples)
    # Batch computation of skewness differences, default is 0 -> sample_skew-0
    skew_diff = torch.norm(sample_skew, dim=1)  

    # Comprehensive evaluation: each component can be weighted as needed
    score = mean_diff + std_diff + 0.1 * skew_diff
    return score

def select_algorithm_batch(n_trials, n_samples, mean, std, device, seed):
    if seed is not None:
        torch.manual_seed(seed)
    # Batch computation, where n_trials indicates the batch size.
    samples = torch.normal(mean.expand(n_trials, n_samples, -1), std.expand(n_trials, n_samples, -1)).to(device)  # Batch sampling
    scores = evaluate_distribution_batch(samples, mean, std)  # Batch evaluating
    
    best_score, best_idx = torch.min(scores, dim=0)  # Find smaples with best (small) scores
    worst_score, worst_idx = torch.max(scores, dim=0)
    best_sample = samples[best_idx]  # Get best samples
    worst_sample = samples[worst_idx]
    
    return best_sample, worst_sample, best_score, worst_score



def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train Stable Diffusion XL for InstructPix2Pix."
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default="ds_inf/imagenet_1k_mapping.json",
    )
    parser.add_argument("--pretrained_path", default='/scratch/zhao.lin1/DiT-XL-2-256', type=str)
    parser.add_argument("--save_dir", default='/scratch/zhao.lin1/distilled_images/', type=str)
    parser.add_argument("--statistic_path", default='/scratch/zhao.lin1/ddim_inversion_statistic', type=str)
    parser.add_argument(
        "--start",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--ipc",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=18,
    )
    parser.add_argument(
        "--i_step",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--m",
        type=int,
        default=100000,
    )
    args = parser.parse_args()
    
    return args


def main():
    args = parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.set_device(device)
    
    # x5_step i=[3, 8, 13, 18, 23, 28, 33, 38, 43, 48] start_step = [45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    # 10-20_step i=[24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39] start_step = [24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]

    wnid_to_index = load_mapping(args.mapping_file)
    class_dirs = sorted(list(wnid_to_index.keys()))[args.start:args.end]



    pipe = DiTPipeline.from_pretrained(args.pretrained_path, torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config) 
    pipe = pipe.to(device)

    for class_dir in tqdm(class_dirs):
        # Compute the best and worst samples
        statics = torch.load(os.path.join(args.statistic_path, class_dir+'.pt'))
        mean = statics['mean'][args.i_step].to(device)
        variance = statics['variance'][args.i_step].to(device)
        std = torch.sqrt(variance)

        latents_best = None
        latents_worst = None
        best_overall_score = float('inf')  # initialize as inf
        worst_overall_score = float('-inf')  # initialize as -ing

        # group sampling
        for i in range(args.m//10):
            seed = i * 12345
            best_sample, worst_sample, best_score, worst_score = select_algorithm_batch(10000, args.ipc, mean, std, device, seed)
            
            # Update best and worst samples
            if best_score < best_overall_score:
                best_overall_score = best_score
                latents_best = best_sample
            
            if worst_score > worst_overall_score:
                worst_overall_score = worst_score
                latents_worst = worst_sample

        # Output results
        print("Best overall score:", best_overall_score)
        print("Worst overall score:", worst_overall_score)
        

        latents_best = latents_best.view(-1,4,32,32)
        # latents_worst = latents_worst.view(-1,4,32,32)


        # Generate images
        for k, latent in enumerate(latents_best):            
            image = sample(
                pipe,
                class_labels=torch.tensor(wnid_to_index[class_dir]).unsqueeze(0),
                start_latents=latent.unsqueeze(0).to(torch.float16),
                start_step=args.start_step,
                num_inference_steps=50,
                device=device
            )
            os.makedirs(os.path.join(args.save_dir,class_dir.split('/')[-1]), exist_ok=True)
            pipe.numpy_to_pil(image)[0].resize((224, 224), Image.LANCZOS).save(os.path.join(args.save_dir,class_dir.split('/')[-1], str(k)+'.png'))
            


if __name__ == "__main__":
    main()

