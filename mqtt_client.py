import os
import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, client_id):

        self.broker_address = os.getenv('BROKER_ADDRESS')
        self.broker_port = int(os.getenv('BROKER_PORT'))
        self.client_id = client_id
        self.username = os.getenv('BROKER_USER')
        self.password = os.getenv('BROKER_PASSWORD')

        # Set up MQTT client instance
        self.client = mqtt.Client(client_id)

        # Set up MQTT callback functions
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        # Set up MQTT client credentials
        self.client.username_pw_set(self.username, self.password)

    # Connect to the MQTT broker
    def connect(self):
        self.client.connect(self.broker_address, self.broker_port)
        # self.client.loop_start()

    # Disconnect from the MQTT broker
    def disconnect(self):
        # self.client.loop_stop()
        self.client.disconnect()

    # Define MQTT callback function for when the client 
    # receives a CONNACK response from the server
    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        # client.subscribe("test/topic")

    # Define MQTT callback function for when the client 
    # publishes a message
    def on_message(self, client, userdata, msg):
        print(msg.topic + " " + str(msg.payload))

    # Publish a message
    def publish(self, topic, payload):
        self.client.publish(topic, payload)

    # Subscribe to a topic
    # def subscribe(self, topic):
    #     self.client.subscribe(topic)
