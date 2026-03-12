import matplotlib.pyplot as plt

# 에포크별 Train 및 Validation Loss

train_loss = [
    0.4533, 0.4228, 0.4138, 0.4081, 0.4041,
    0.4008, 0.3982, 0.3960, 0.3941, 0.3921,
    0.3906, 0.3892, 0.3881, 0.3872, 0.3864,
    0.3857, 0.3851, 0.3846, 0.3841
]

val_loss = [
    0.3993, 0.3916, 0.3880, 0.3857, 0.3843,
    0.3833, 0.3826, 0.3821, 0.3817, 0.3814,
    0.3811, 0.3809, 0.3807, 0.3806, 0.3805,
    0.3804, 0.3803, 0.3802, 0.3801
]


length = len(train_loss)  # Assuming both lists are of the same length
epochs = list(range(1, length + 1))

plt.figure()
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.show()