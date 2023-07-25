import cv2
from minio import Minio
import tempfile

def upload_image_to_minio(img_path, img_id, frame_id, bucket_name, object_prefix, minio_config):
    # Read image from disk
    img = cv2.imread(img_path)
    
    # Check if the image has been loaded correctly
    if img is None:
        print("Image not loaded correctly.")
        return

    # Create a temporary file
    temporal = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temporal.name, img)

    # Connect to Minio server
    minio_client = Minio(
        minio_config['host'], 
        access_key=minio_config['access_key'], 
        secret_key=minio_config['secret_key'], 
        secure=minio_config['secure']
    )

    # Define the object name
    object_name = object_prefix + str(img_id) + "_" + str(frame_id) + ".jpg"

    # Upload the image
    minio_client.fput_object(bucket_name, object_name, temporal.name)

    # Close the temporary file
    temporal.close()

# Define your Minio configuration
minio_config = {
    'host': "minio-cli.platform.flexigrobots-h2020.eu",
    'access_key': "Pilot1-FlexiGroBotsH2020",
    'secret_key': "!&MXsK30%3Aze1$",
    'secure': True,
}

# Use the function to upload an image
upload_image_to_minio(
    img_path='inputs/stitch_8_2/000000.png', 
    img_id=123, 
    frame_id=456, 
    bucket_name='super-scenario-p1', 
    object_prefix='data/botrytis/images/', 
    minio_config=minio_config
)
