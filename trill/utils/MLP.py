# Taken straight from TemStaPro!

from torch import nn

class MLP_C2H2(nn.Module):
    def __init__(self, input_size=None, hidden_size_1=None, hidden_size_2=None):
        if(input_size == None):
            self.input_size = 1280
        else:
            self.input_size = int(input_size)

        if(hidden_size_1 == None):
            self.hidden_size_1 = 640
        else:
            self.hidden_size_1 = int(hidden_size_1)

        if(hidden_size_2 == None):
            self.hidden_size_2 = 320
        else:
            self.hidden_size_2 = int(hidden_size_2)

        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size_1),
            nn.ReLU(),
            nn.Linear(self.hidden_size_1, self.hidden_size_2),
            nn.ReLU(),
            nn.Linear(self.hidden_size_2, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


def inference_epoch(model, test_loader, device="cpu"):
    """
    From TemStaPro repo 
    """
    inferences = {}
    for i, data in enumerate(test_loader, 0):
        inputs = data[0][0]
        targets = data[1][0]
        inputs = inputs.to(device)
        outputs = model(inputs.float())
        outputs = outputs.detach().cpu().numpy()
        
        seq_id = targets
        
        for output in outputs:
            inferences[seq_id] = output

    return inferences

def prepare_inference_dictionaries(sequences_list):
    """
    From TemStaPro repo 
    """
    averaged_inferences = []
    binary_inferences = []
    labels = []
    clashes = [] 

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
    
    return (averaged_inferences, binary_inferences, labels)