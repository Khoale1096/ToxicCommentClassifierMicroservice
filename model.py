import asyncio
from PIL import Image
from ToxicComment.ToxicCommentClassifier import ToxicCommentClassifier

model = None

async def init():
    """
    This method will be run once on startup. You should check if the supporting files your
    model needs have been created, and if not then you should create/fetch them.
    """
    await asyncio.sleep(2)
    global model

    print('Loading ToxicCommentClassifier model')
    model = ToxicCommentClassifier()
    model.load_labels_values()
    model.load_model


def predict(image_file):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given a file-like object
    with the image as an input.
    """

    image = Image.open(image_file.name, mode='r')

    return {
        "someResultCategory": "actualResultValue",
    }


def predict_text(text_sentences):
    """
    Interface method between model and server. This signature must not be
    changed and your model must be able to predict given a list of strings 
    as an input.
    """
    global model
    if model == None:
        raise RuntimeError("ToxicCommentClassifier model is not loaded properly")

    output = model.predict([text_sentences])

    return output