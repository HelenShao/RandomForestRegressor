import numpy as np
from sklearn.ensemble import RandomForestRegressor
import optuna
import data 

################################# Objective Function #############################
class objective(object):
    def __init__(self, f_rockstar, device, seed, n_esti_min, 
                 n_esti_max, leaf_min, leaf_max): 
        
        self.f_rockstar  = f_rockstar
        self.device      = device
        self.seed        = seed 
        self.n_esti_min  = n_esti_min
        self.n_esti_max  = n_esti_max
        self.leaf_min    = leaf_min
        self.leaf_max    = leaf_max
        
    
    def __call__(self, trial):

        # Generate the model using trial suggestions
        n_estimators = trial.suggest_int("n_estimators", n_esti_min, n_esti_max)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", leaf_min, leaf_max)
        
        model = RandomForestRegressor(n_estimators = n_estimators,
                              criterion='mse', min_samples_leaf = min_samples_leaf)

        # Load the datasets (normalized)
        train_x = np.load("norm_train_x.npy")
        train_y = np.load("norm_train_y.npy")
        
        valid_x = np.load("norm_valid_x.npy")
        valid_y = np.load("norm_valid_y.npy")
        
        test_x = np.load("norm_test_x.npy")
        test_y = np.load("norm_test_y.npy")

        # Train the model: predict V_rms from the other 9 properties
        n_halos = train_x.shape[0]
        model.fit(train_x, train_y.reshape(n_halos,))

        # Validation of the model
        min_valid = 1e40
        count, loss_valid = 0, 0.0
        for i in range(len(valid_x)):
            pred = model.predict(valid_x[i].reshape(1, -1))
            loss_valid = np.mean((pred-valid_y[i])**2)

        if loss_valid<min_valid:  
            min_valid = loss_valid

        return min_valid

##################################### INPUT #######################################
# Data Parameters
# n_halos      = 3674
# n_properties = 9
seed         = 4
# mass_per_particle = 6.56561e+11 
f_rockstar = "Rockstar_z=0.0.txt"

# Model Parameters
n_esti_min = 10    # Minimum number of trees
n_esti_max = 300   # Maximum number of trees
leaf_min = 1       # Minimum number of samples before split
leaf_max = 200     # Maximum number of samples before split

# Optuna Parameters
n_trials   = 2000 
study_name = 'Halos_RFR_params'
n_jobs     = 1

############################## Start OPTUNA Study ###############################

# Use GPUs if avaiable
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')

if __name__ == "__main__":
    
    # define the optuna study and optimize it
    objective = objective(f_rockstar, device, seed, n_esti_min, 
                          n_esti_max, leaf_min, leaf_max)
    
    # Optimization direction = minimize valid_loss
    sampler = optuna.samplers.TPESampler(n_startup_trials=300) 
    study = optuna.create_study(study_name=study_name, sampler=sampler,direction="minimize", load_if_exists = True)
    study.optimize(objective, n_trials=n_trials, n_jobs = n_jobs)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Print parameters of the best trial
    trial = study.best_trial
    print("Best trial: number {}".format(trial.number))
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
