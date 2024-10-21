import torch
import torch.nn as nn
import learn2learn as l2l

class GridTransformModel(nn.Module):
    def __init__(self):
        super(GridTransformModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Add more layers as needed
        )
        self.decoder = nn.Sequential(
            # Up-convolution or transpose convolution layers
            nn.ConvTranspose2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  # Assuming output values between 0 and 1
        )

    def forward(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output



# Initialize the model and MAML wrapper
model = GridTransformModel()
maml = l2l.algorithms.MAML(model, lr=fast_lr, first_order=False)

# Define the optimizer
meta_optimizer = torch.optim.Adam(maml.parameters(), lr=meta_lr)

# Training loop
for iteration in range(num_iterations):
    meta_optimizer.zero_grad()
    task_losses = []
    for _ in range(batch_size):
        # Sample a task
        task = generate_task()
        data = task['data']

        # Split into support and query sets
        support_set = data[:num_support]
        query_set = data[num_support:]

        # Create a clone of the model for adaptation
        learner = maml.clone()

        # Adaptation on support set
        for input_grid, output_grid in support_set:
            input_tensor = torch.tensor(input_grid).unsqueeze(0).unsqueeze(0).float()
            output_tensor = torch.tensor(output_grid).unsqueeze(0).unsqueeze(0).float()
            pred = learner(input_tensor)
            loss = loss_fn(pred, output_tensor)
            learner.adapt(loss)

        # Evaluation on query set
        query_losses = []
        for input_grid, output_grid in query_set:
            input_tensor = torch.tensor(input_grid).unsqueeze(0).unsqueeze(0).float()
            output_tensor = torch.tensor(output_grid).unsqueeze(0).unsqueeze(0).float()
            pred = learner(input_tensor)
            query_loss = loss_fn(pred, output_tensor)
            query_losses.append(query_loss)

        # Compute mean query loss for the task
        task_loss = torch.stack(query_losses).mean()
        task_losses.append(task_loss)

    # Meta-update
    meta_loss = torch.stack(task_losses).mean()
    meta_loss.backward()
    meta_optimizer.step()

    if iteration % print_every == 0:
        print(f"Iteration {iteration}, Meta Loss: {meta_loss.item()}")
