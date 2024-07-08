from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=57600, help='frequency of saving the latest results')
        parser.add_argument('--print_freq', type=int, default=2880, help='frequency of ploting losses')
        parser.add_argument('--save_epoch_freq', type=int, default=40, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # training parameters
        parser.add_argument('--n_epochs_joint', type=int, default=400, help='number of epochs with the initial learning rate')

        parser.add_argument('--lr_joint', type=float, default=1e-4, help='initial learning rate')


        
          
        self.isTrain = True
        return parser
