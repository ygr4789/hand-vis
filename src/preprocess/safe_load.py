import torch

def safe_load_pkl(path):
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	
	try:
		data = torch.load(path, map_location=device, weights_only=False)
	except RuntimeError as e:
		print(f"Standard load failed due to nested serialization: {e}")
		# fallback for nested torch.load inside pickle
		import pickle
		import io

		class CPU_Unpickler(pickle.Unpickler):
			def find_class(self, module, name):
				if module == 'torch.storage' and name == '_load_from_bytes':
					def load_from_bytes(b):
						return torch.load(io.BytesIO(b), map_location=device)
					return load_from_bytes
				return super().find_class(module, name)

		with open(path, 'rb') as f:
			data = CPU_Unpickler(f).load()
	return data