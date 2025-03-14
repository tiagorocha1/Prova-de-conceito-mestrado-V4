import cv2
import asyncio
import io
from datetime import datetime
from minio_utils import save_image_to_minio
from rabbitmq_manager import rabbitmq_manager
import os

# Configuração do intervalo de captura (em segundos)
CAPTURE_INTERVAL = int(os.getenv("CAPTURE_INTERVAL", 1))  # Padrão: 1 segundo

class WebcamCapture:
    """ Captura frames da webcam e envia para MinIO + RabbitMQ """

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.running = False

    async def capture_and_upload(self):
        """ Captura frames da webcam, salva no MinIO e envia mensagem ao RabbitMQ. """
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"❌ Erro: A câmera {self.camera_index} não pôde ser aberta.")
            return

        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Erro ao capturar frame.")
                    continue

                # Obtém a data atual no formato DD-MM-AAAA
                current_date = datetime.now().strftime("%d-%m-%Y")
                timestamp = str(int(datetime.now().timestamp() * 1000))

                # Nome do arquivo com subpasta da data corrente
                object_name = f"{current_date}/{timestamp}.png"
                minio_path = f"{object_name}"

                # Converte para buffer PNG
                _, buffer = cv2.imencode(".png", frame)
                image_buffer = io.BytesIO(buffer.tobytes())

                # Salva no MinIO
                await asyncio.to_thread(save_image_to_minio, image_buffer, object_name)

                # Envia mensagem ao RabbitMQ
                await rabbitmq_manager.send_message(minio_path)

                print(f"✅ Imagem salva e mensagem enviada: {minio_path}")

                # Espera 1 segundo antes de capturar o próximo frame
                await asyncio.sleep(CAPTURE_INTERVAL)

        except KeyboardInterrupt:
            print("⏹️ Interrompido pelo usuário.")

        finally:
            cap.release()
            cv2.destroyAllWindows()

