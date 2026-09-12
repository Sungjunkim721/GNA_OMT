Place the pretrained OMT-mixer checkpoint here as:

    checkpoints/OMT_Mixer.pth

The manuscript experiments used a PyTorch state dictionary saved with:

    torch.save(forward_model.state_dict(), forward_model_name)
