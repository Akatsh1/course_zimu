from modelscope import snapshot_download
#模型下载
model_dir = snapshot_download('pengzhendong/faster-whisper-large-v3-turbo',local_dir='./model')