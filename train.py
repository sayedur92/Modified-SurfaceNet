import sys
from pathlib import Path


sys.path.append('/home/hpc/vlgm/vlgm103v/genmatpro/idea/')

from accelerate import Accelerator
from accelerate.tracking import TensorBoardTracker

from my_datasets.surface_dataset import PicturesDataset, SurfaceDataset
from trainers.trainer import Trainer
from utils.parser import parse

if __name__ == '__main__':
    
    args = parse()

    
    tracker = TensorBoardTracker(run_name=args.tag, logging_dir=args.logdir)
    accelerator = Accelerator(log_with=tracker)
    accelerator.init_trackers("surfacenet")

    
    train_path = Path(args.dataset) / 'trainBlended'
    test_path = Path(args.dataset) / 'testBlended'
    sketch_train_path = Path(args.dataset) / 'trainSketch'
    sketch_test_path = Path(args.dataset) / 'testSketch'

    # Create datasets
    dset_train = SurfaceDataset(
        dset_dir=train_path,
        sketch_dir=sketch_train_path,  
        load_size=args.resize
    )

    dset_test = SurfaceDataset(
        dset_dir=test_path,
        sketch_dir=sketch_test_path,  
        load_size=args.resize
    )


    datasets = {
        'train': {'synth': dset_train},
        'test': {'synth': dset_test},
    }
    

    
    if args.train_real:
        datasets['train']['real'] = PicturesDataset(
            dset_dir=train_path,
            sketch_dir=None,  
            load_size=args.resize
        )
        datasets['test']['real'] = PicturesDataset(
            dset_dir=test_path,
            sketch_dir=None,  
            load_size=args.resize
        )

    
    trainer = Trainer(args, accelerator, tracker, datasets)
    trainer.train()
    accelerator.end_training()
