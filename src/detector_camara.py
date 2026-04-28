import cv2

# 0 es usualmente la cámara integrada de la laptop
cap = cv2.VideoCapture(0)

print("Presiona 'q' para cerrar la cámara")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mostrar el video en una ventana
    cv2.imshow('Prueba de Camara - Proyecto Frutas', frame)

    # Si presionas la tecla 'q', se cierra
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()