import torch

class ModelSaver:
    def __init__(self, save_dir, loss):
        """
        Initialize the ModelSaver to track minimum loss_rec for specific kld intervals.

        Args:
            intervals (list of tuples): List of (start, end) tuples representing kld intervals.
            save_dir (str): Directory to save the model checkpoints.
        """
        
        self.loss = loss
        if self.loss == "tcvae":
            self.intervals = [0,1,2,3,4,5,6,7,8,9,10] # ranges for total correlation loss
            self.loss_name = "tc"
        elif self.los == "bvae":
            self.intervals = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1] # ranges for kld loss
            self.loss_name = "kld"
            
        self.save_dir = save_dir + "/"
        self.min_loss_rec = [1 for i in range(len(self.intervals))] # list to store best rec value in each interval

    def save_model_if_new_min(self, model, loss_rec, loss_kld_tc):
        """
        Save the model if loss_rec or is a new minimum in one of the intervals for kld / total correlation loss.
        """
        
        saved_flag = False  # Flag to indicate if a model was saved in this epoch
        for i in range(len(self.intervals)-1):
            if self.intervals[i] <= loss_kld_tc < self.intervals[i+1]:
                if loss_rec < self.min_loss_rec[i]:
                    # Update the minimum loss_rec
                    self.min_loss_rec[i] = loss_rec

                    # Save the model
                    model_name = f"model_{self.loss_name}_{i}_{i+1}.pth"
                    save_path = self.save_dir + model_name
                    torch.save(model.state_dict(), save_path)
                    saved_flag = True
                break
                
        # Log to the file
        if saved_flag:
            with open(self.save_dir + "model_values.txt", "a") as f:
                f.write(f"{model_name}, Loss_rec: {loss_rec:.6f}, Loss_{self.loss_name}: {loss_kld_tc:.6f}\n")
        
        return saved_flag
