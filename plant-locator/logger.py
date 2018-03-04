import visdom
import torch
import numbers


class Logger():
    def __init__(self, env_name='Logger Env'):

        # Visdom setup
        self.client = visdom.Visdom(env=env_name)

        # Each of the 'windows' in visdom web panel
        self.viz_train_input_win = None
        self.viz_train_loss_win = None
        self.viz_train_gt_win = None
        self.viz_train_est_win = None
        self.viz_val_input_win = None
        self.viz_val_loss_win = None
        self.viz_val_gt_win = None
        self.viz_val_est_win = None

    def train_losses(self, terms, iteration_number, terms_legends=None):
        """Plot a new point of the training losses (scalars) to Visdom.
        All losses will be plotted in the same figure/window.

        :param terms: List of scalar losses.
                      Each element will be a different plot in the y axis.
        :param iteration_number: Value of the x axis in the plot.
        :param terms_legends: Legend of each term.
        """

        # Watch dog
        if terms_legends is not None and \
                len(terms) != len(terms_legends):
            raise ValueError('The number of "terms" and "terms_legends" must be equal, got %s and %s, respectively'
                             % (len(terms), len(terms_legends)))
        if not isinstance(iteration_number, numbers.Number):
            raise ValueError('iteration_number must be a number, got %s'
                             % iteration_number)

        y = torch.cat(terms).view(1, -1).data.cpu()
        x = torch.Tensor([iteration_number]).repeat(1, len(terms))
        if terms_legends is None:
            terms_legends = ['Term %s' % t
                             for t in range(1, len(terms) + 1)]

        # Send training loss to Visdom
        self.win_train_loss = \
            self.client.updateTrace(Y=y,
                                    X=x,
                                    opts=dict(title='Training',
                                              legend=terms_legends,
                                              ylabel='Loss',
                                              xlabel='Iteration'),
                                    append=True,
                                    win='0')
        if self.win_train_loss == 'win does not exist':
            self.win_train_loss = \
                self.client.line(Y=y,
                                 X=x,
                                 opts=dict(title='Training',
                                           legend=terms_legends,
                                           ylabel='Loss',
                                           xlabel='Iteration'),
                                 win='0')

    def image(self, imgs, titles, windows):
        """Send images to Visdom.
        Each image will be shown in a different window/plot.

        :param imgs: List of numpy images.
        :param titles: List of titles of each image.
        :param windows: List of window names.
        """

        # Watchdog
        if not(len(imgs) == len(titles) == len(windows)):
            raise ValueError('The number of "imgs", "titles" and '
                             '"windows" must be equal, got '
                             '%s, %s and %s, respectively'
                             % (len(imgs), len(titles), len(windows)))

        for img, title, win in zip(imgs, titles, windows):
            self.client.image(img,
                              opts=dict(title=title),
                              win=str(win))

    def val_losses(self, terms, iteration_number, terms_legends=None):
        """Plot a new point of the training losses (scalars) to Visdom.  All losses will be plotted in the same figure/window.

        :param terms: List of scalar losses.
                      Each element will be a different plot in the y axis.
        :param iteration_number: Value of the x axis in the plot.
        :param terms_legends: Legend of each term.
        """

        # Watchdog
        if terms_legends is not None and \
                len(terms) != len(terms_legends):
            raise ValueError('The number of "terms" and "terms_legends" must be equal, got %s and %s, respectively'
                             % (len(terms), len(terms_legends)))
        if not isinstance(iteration_number, numbers.Number):
            raise ValueError('iteration_number must be a number, got %s'
                             % iteration_number)

        y = torch.stack(terms).view(1, -1).data.cpu()
        x = torch.Tensor([iteration_number]).repeat(1, len(terms))
        if terms_legends is None:
            terms_legends = ['Term %s' % t for t in range(1, len(terms) + 1)]

        # Send validation loss to Visdom
        self.win_val_loss = \
            self.client.updateTrace(Y=y,
                                    X=x,
                                    opts=dict(title='Validation',
                                              legend=terms_legends,
                                              ylabel='Loss',
                                              xlabel='Epoch'),
                                    append=True,
                                    win='4')
        if self.win_val_loss == 'win does not exist':
            self.win_val_loss = \
                self.client.line(Y=y,
                                 X=x,
                                 opts=dict(title='Validation',
                                           legend=terms_legends,
                                           ylabel='Loss',
                                           xlabel='Epoch'),
                                 win='4')
