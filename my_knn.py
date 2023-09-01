import torch
# Make sure your CUDA is available.
assert torch.cuda.is_available()
import faiss


def get_hypergraph(data,k):
    if faiss.get_num_gpus() == 0:
        print("No GPU available, switching to CPU")
        res = faiss.StandardGpuResources()
    else:
        print("GPU available, using GPU")
        res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, 300)
    # Add data to the index
    index.add(data)
    # Search for k-nearest neighbors
    k = k+1
    query = data
    distance, indices = index.search(query, k)
    # not include itself
    indices = indices[:,1:]
    # Retrieve the neighbors
    return indices


def get_neg_hypergraph(data,k):
    if faiss.get_num_gpus() == 0:
        print("No GPU available, switching to CPU")
        res = faiss.StandardGpuResources()
    else:
        print("GPU available, using GPU")
        res = faiss.StandardGpuResources()
    index = faiss.GpuIndexFlatIP(res, 300)
    # Add data to the index
    index.add(-1*data)
    # Search for k-nearest neighbors
    k = k
    query = data
    distance, indices = index.search(query, k)
    # Retrieve the neighbors
    return indices

def get_initial_value(model,data):
    with torch.no_grad():
        model.eval()
        x_bar, hidden = model.ae(data)
        torch.cuda.empty_cache()
    return x_bar, hidden


