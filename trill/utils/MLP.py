# Taken straight from TemStaPro!

from torch import nn

class MLP_C2H2(nn.Module):
    def __init__(self,
        input_size=1024,
        hidden_size_1=512,
        hidden_size_2=256
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2

        self.model = nn.ModuleList(
            [
                nn.Linear(self.input_size, self.hidden_size_1),
                nn.ReLU(),
            ]
            + [
                nn.Linear(self.hidden_size_1, self.hidden_size_2),
                nn.ReLU(),
            ]
            + [
                nn.Linear(self.hidden_size_2, 1),
                nn.Sigmoid()
            ]
        )
        self.loss_function = nn.BCELoss()

    def forward(self, point):
        for layer in self.model:
            point = layer(point)
        return point

    def calculate_loss(self, point, label):
        return self.loss_function(point, label)



def prepare_inference_dictionaries(sequences_list, is_npz=False):
	"""
	Initialising dictionaries to save inferences.

	sequences_lists - LIST of dictionaries with information about sequences
	is_npz - BOOLEAN that indicates whether an NPZ file or a FASTA file is 
		processed

	returns (LIST, LIST, LIST, LIST)
	"""
	averaged_inferences = []
	binary_inferences = []
	labels = []
	clashes = [] 

	if(is_npz):
		averaged_inferences.append({})
		binary_inferences.append({})
		labels.append({})
		clashes.append({})
		for seq in sequences_list[0]:
			averaged_inferences[0][seq[0].split("|")[1]] = []
			binary_inferences[0][seq[0].split("|")[1]] = []
			labels[0][seq[0].split("|")[1]] = []
			clashes[0][seq[0].split("|")[1]] = []
	else:
		for i, seq_dict in enumerate(sequences_list):
			if(seq_dict is None): break
			averaged_inferences.append({})
			binary_inferences.append({})
			labels.append({})
			clashes.append({})
			for seq in seq_dict.keys():
				averaged_inferences[i][seq] = []
				binary_inferences[i][seq] = []
				labels[i][seq] = []
				clashes[i][seq] = []
	
	return (averaged_inferences, binary_inferences, labels, clashes)

def inference_epoch(model, test_loader, identifiers=[], device="cpu"):
	"""
	Making inferences for each given protein sequence.

	model - torch.nn.Module with a defined architecture
	test_loader - DataLoader with a dataset loaded for inferences
	identifiers - LIST with sequence identifiers used as keys in inferences DICT
	device - STRING that determines the processor used

	returns DICT with inferences 
	"""
	inferences = {}
	for i, data in enumerate(test_loader, 0):
		inputs, targets = data
		# inputs, targets = inputs.to(device), targets.to(device)
		inputs = inputs[0].to(device)
		outputs = model(inputs.float())
		outputs = outputs.detach().cpu().numpy()
		
		# seq_id = identifiers[i]
		seq_id = targets[0]
		inferences[seq_id] = outputs[0]
		# for output in outputs: 
		# 	inferences[seq_id] = output[0]

	return inferences