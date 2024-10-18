import tf2onnx
import onnx
from onnx2pytorch import ConvertModel
import keras


loaded_model = keras.models.load_model('aidetector.keras')
onnx_model, _= tf2onnx.convert.from_keras(loaded_model)
pytorch_model = ConvertModel(onnx_model)

print(pytorch_model)
