import torch
import torch.nn as nn

# # Create a 2D input tensor with 32 samples and 300 input features
# input_tensor = torch.randn(32, 300)

# # Create a linear layer that maps 300 input features to 64 output features
# linear_layer = nn.Linear(300*32 +8, 64)

# print(torch.randn(72).size())
# emb_layer = nn.Embedding(72,8)
# emb = emb_layer(torch.zeros(1, dtype=torch.int32))
# print(emb.size())
# emb = emb_layer(torch.zeros(1, dtype=torch.int32)).squeeze(1)
# print(emb.size())

# input_tensor = torch.cat((input_tensor,emb),1)
# print(input_tensor.size())
# # Apply the linear transformation to the input tensor
# output_tensor = linear_layer(input_tensor)

# # Print the shape of the output tensor
# print(output_tensor.shape)  # Should print torch.Size([32, 64])

ori_emb = nn.Embedding(72,8)
output = ori_emb(torch.zeros(1,dtype=torch.int32))
print(output)
output = ori_emb(torch.zeros(1,dtype=torch.int32)).squeeze(1)
print(output)