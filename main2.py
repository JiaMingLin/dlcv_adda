import click
import params
from core import train_src, train_tgt, evaluation
from models import Discriminator, LeNetClassifier, LeNetEncoder
from utils import get_data_loader, init_model, init_random_seed

exp_list = ['MNIST_USPS', 'USPS_MNISTM', 'MNISTM_SVHN', 'SVHN_USPS']
cases = ['source', 'adapt', 'target']

@click.command()
@click.option('--exp', type=click.Choice(exp_list), required=True)
@click.option('--case', type=click.Choice(cases), default='adapt')
@click.option('--affine', is_flag=True)
@click.option('--num_epochs', type=int, default=200)
def experiments(exp, case, affine, num_epochs):
    
    print(exp, case, affine, num_epochs)
    
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset
    src_dataset, tgt_dataset = exp.split('_')
    src_data_loader = get_data_loader(src_dataset)
    src_data_loader_eval = get_data_loader(src_dataset, train=False)
    
    tgt_data_loader = get_data_loader(tgt_dataset)
    tgt_data_loader_eval = get_data_loader(tgt_dataset, train=False)

    # load models
    src_encoder = init_model(
        net=LeNetEncoder(),
        restore=params.src_encoder_restore,
        exp = exp
    )
    src_classifier = init_model(
        net=LeNetClassifier(),
        restore=params.src_classifier_restore,
        exp = exp
    )
    tgt_encoder = init_model(
        net=LeNetEncoder(),
        restore=params.tgt_encoder_restore,
        exp = exp
    )
    critic = init_model(
        Discriminator(
            input_dims=params.d_input_dims,
            hidden_dims=params.d_hidden_dims,
            output_dims=params.d_output_dims
        ),
        exp =  exp,
        restore=params.d_model_restore)
    
    # train source model
    print("=== Training classifier for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source Classifier <<<")
    print(src_classifier)
    
    if not (src_encoder.restored and src_classifier.restored and params.src_model_trained):
        src_encoder, src_classifier = train_src(
            exp, src_encoder, src_classifier, src_data_loader, src_data_loader_eval)
    
    # eval source model
    print("=== Evaluating classifier for source domain ===")
    evaluation(src_encoder, src_classifier, src_data_loader_eval)
    
    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)
    
    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())
    
    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt(exp, src_encoder, tgt_encoder, critic, src_classifier,
                                src_data_loader, tgt_data_loader, tgt_data_loader_eval)
        
    # eval target encoder on test set of target dataset
    print("=== Evaluating classifier for encoded target domain ===")
    print(">>> source only <<<")
    evaluation(src_encoder, src_classifier, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    evaluation(tgt_encoder, src_classifier, tgt_data_loader_eval)

if __name__ == '__main__':
    experiments()