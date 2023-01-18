##### HELPER FUNCTION FOR FEATURE EXTRACTION: https://github.com/kozodoi/website/blob/master/_notebooks/2021-05-27-extracting-features.ipynb
def get_features(name, dict): #dict is dictionary for saving features, should be called "features"
    def hook(model, input, output):
        dict[name] = output.detach()
    return hook