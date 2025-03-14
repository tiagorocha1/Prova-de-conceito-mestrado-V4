import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import asyncio
from threading import Thread
from webcam_capture import WebcamCapture

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Captura de Webcam")
        self.root.geometry("400x300")

        # Lista de webcams disponíveis
        self.cameras = self.get_available_cameras()

        # Dropdown para selecionar a webcam
        self.camera_label = tk.Label(root, text="Selecione a Webcam:")
        self.camera_label.pack(pady=10)

        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(root, textvariable=self.camera_var, values=self.cameras)
        self.camera_dropdown.pack()
        if self.cameras:
            self.camera_dropdown.current(0)

        # Botão para iniciar a captura
        self.start_button = tk.Button(root, text="Iniciar Captura", command=self.start_capture, bg="green", fg="white")
        self.start_button.pack(pady=20)

        # Botão para parar a captura
        self.stop_button = tk.Button(root, text="Parar Captura", command=self.stop_capture, bg="red", fg="white")
        self.stop_button.pack()
        self.stop_button.config(state=tk.DISABLED)

        self.capture_instance = None

        # Criar um loop `asyncio` dentro do Tkinter
        self.loop = asyncio.new_event_loop()
        self.thread = Thread(target=self.run_asyncio_loop, daemon=True)
        self.thread.start()

    def get_available_cameras(self):
        """ Descobre quais webcams estão disponíveis no sistema """
        available_cameras = []
        for i in range(5):  # Testamos até 5 câmeras disponíveis
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(f"Camera {i}")
                cap.release()
        return available_cameras if available_cameras else []

    def start_capture(self):
        """ Inicia a captura da webcam selecionada """
        if not self.cameras:
            messagebox.showerror("Erro", "Nenhuma câmera encontrada!")
            return

        camera_index = int(self.camera_var.get().split()[-1])  # Obtém o índice da câmera
        self.capture_instance = WebcamCapture(camera_index)
        self.capture_instance.running = True

        # Executar a captura de forma assíncrona
        asyncio.run_coroutine_threadsafe(self.capture_instance.capture_and_upload(), self.loop)

        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def stop_capture(self):
        """ Para a captura da webcam """
        if self.capture_instance:
            self.capture_instance.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def run_asyncio_loop(self):
        """ Inicia um loop asyncio separado para executar tarefas assíncronas dentro do Tkinter """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    root.mainloop()
