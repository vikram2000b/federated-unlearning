import torch

def apply_dampening(global_model, forget_client_contributions, retain_clients_contirbutions, dampening_constant,
                    dampening_upper_bound, ratio_cutoff):
        """
        Apply dampening to the global model based on the gradients of the local models.
        Args:
            global_model: The global model which will be dampened. 
            forget_client_contributions: The gradient contributions of the forget clients/models.
            retain_clients_contributions: The gradient contributions of the retain clients/models.
            dampening_constant: The dampening constant.
            dampening_upper_bound: The upper bound for the final dampening factor. Used to cap the increasing of 
            the parameters.
            ratio_cutoff: The cutoff/filter factor for ratios. Any parameter having the ratio greater than this value will not be updated.
              A high ratio means less contribution of the forget model, leading to less dampening. 
        Returns:
            The updated global model.
        """

        with torch.no_grad():
          for (global_name, forget_grads), (index, retain_grads) in zip(
              forget_client_contributions.items(),
              retain_clients_contirbutions.items()
          ):

              if len(forget_grads.shape) > 0:
                # Synapse Dampening with parameter dampening constant
                weight = global_model[global_name]
                # diff = torch.abs(g2_grads - g1_grads) # torch.abs(torch.abs(g2_grads) - torch.abs(g1_grads))
                retain_contribution = torch.abs(retain_grads) # epsilon
                forget_contribution = torch.abs(forget_grads)
                ratio = retain_contribution / forget_contribution
                update_locations = (ratio < ratio_cutoff)
                dampening_factor = torch.mul(ratio, dampening_constant)

                update = dampening_factor[update_locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = (update > dampening_upper_bound)
                update[min_locs] = dampening_upper_bound
                weight[update_locations] = weight[update_locations].mul(update)
        return global_model