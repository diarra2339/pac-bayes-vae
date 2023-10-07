import matplotlib.pyplot as plt

from data import Circle, TwoGaussians
from vae import LipschitzVAE
from utils import heat_scatter
from pac_bayes import recons_bound_diam

epochs = 400
train_size = 50000
val_size = 20000
test_size = 20000
lamda = val_size / 1


if __name__ == '__main__':
    tg = TwoGaussians(centers_x=(-1, 1), sigma=0.1,
                      train_size=train_size, val_size=val_size, test_size=test_size, batch_size=256, workers=7)
    circ = Circle(radius=1.5, max_radius=1.9, sigma=0.1,
                  train_size=train_size, val_size=val_size, test_size=test_size, batch_size=256, workers=7)

    tg_model = LipschitzVAE(encoder_hidden_layers=[100, 100, 100], decoder_hidden_layers=[100, 100, 100], input_dim=2,
                            latent_dim=5, bias=True, config_name='config.json')
    tg_model.train_model(train_loader=tg.train_loader, epochs=epochs, kl_coeff=1, lr=5e-5)
    tg_bound_dico = recons_bound_diam(model=tg_model, val_loader=tg.val_loader, test_loader=tg.test_loader,
                                   lamda=lamda, k_phi=2, k_theta=2, delta=0.05, diameter=tg.diameter)

    c_model = LipschitzVAE(encoder_hidden_layers=[100, 100, 100], decoder_hidden_layers=[100, 100, 100], input_dim=2,
                         latent_dim=5, bias=True, config_name='config.json')
    c_model.train_model(train_loader=circ.train_loader, epochs=epochs, kl_coeff=1, lr=5e-5)
    c_bound_dico = recons_bound_diam(model=c_model, val_loader=circ.val_loader, test_loader=circ.test_loader,
                                   lamda=lamda, k_phi=2, k_theta=2, delta=0.05, diameter=circ.diameter)


    # Plot samples
    tg_samples = tg_model.generate(2000)
    c_samples = c_model.generate(2000)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    heat_scatter(x=tg_samples[:, 0], y=tg_samples[:, 1], sigma=16, r=4, ax=ax1)
    heat_scatter(x=c_samples[:, 0], y=c_samples[:, 1], sigma=16, r=4, ax=ax2)

    # Plot the losses
    tg_model.history.show_losses(['rec_loss', 'kl_div', 'loss'], start_epoch=1)
    c_model.history.show_losses(['rec_loss', 'kl_div', 'loss'], start_epoch=1)


