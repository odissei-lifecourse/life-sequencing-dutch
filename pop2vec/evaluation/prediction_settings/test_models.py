import sys 

from pop2vec.evaluation.prediction_settings.train_and_eval import load_config, create_model


if __name__ == '__main__':
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path)
    model = create_model(256, 7, cfg)
    print(f"{cfg['model_type']} model created")
    print(f"requires grad # = {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print(f"all # = {sum(p.numel() for p in model.parameters())}") 