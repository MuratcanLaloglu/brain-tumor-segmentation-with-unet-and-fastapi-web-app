# Brain Tumor Segmentation with U-Net and FastAPI Web app

This repository contains instructions for training a U-Net model for brain tumor segmentation using Python and Tensorflow. Follow the steps below to set up and train the model.

## Installation

Before running the notebook file, make sure you have installed all the necessary dependencies. You can install them using pip:

```
pip install -r requirements.txt
```


## Training

Once you have installed the requirements, you can proceed with training the U-Net model. Follow these steps:

1. **Run Notebook File**: Open the notebook file (`unet_model_training.ipynb`) in your preferred Python environment (such as Jupyter Notebook or Google Colab).
   
2. **Execute Notebook Cells**: Execute the cells in the notebook sequentially to load the dataset, define the U-Net model architecture, and train the model.

3. **Monitor Training**: Monitor the training process as it progresses. You may want to adjust hyperparameters or experiment with different configurations to achieve the best results.

4. **Save Model**: After training, save the trained model weights for future use or deployment model.

## Serving the Model

Once the model is trained and saved, you can deploy it for inference using FastAPI. Follow these steps:


2. **Run Uvicorn Server**: Start the Uvicorn server by running the following command in your terminal:
```
uvicorn app:app
```


This command will start the server, and your U-Net model will be ready to serve predictions.

## Conclusion

By following these instructions, you should be able to train a U-Net model and deploy it for inference using FastAPI. If you encounter any issues or have questions, feel free to reach out to the repository owner or refer to the documentation provided with the code.

Happy modeling! 🚀
