import os
import pickle
import time
import traceback
import warnings

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from tqdm import tqdm

from utils.build_fully_connected_graph import build_graph_with_knn
from utils.config import parse_args
from utils.config_models import PrecisionMode
from utils.utils_train import create_model, load_class_from_args

warnings.filterwarnings(
    "ignore",
    message=".*The TorchScript type system doesn't support instance-level annotations.*",
    category=UserWarning,
    module="torch.jit._check",
)

MACROS_DIR_NAME = "visualize_macros"
STEPS_TO_RETURN_MULTIPLIER = 100


class SelfFeedError(RuntimeError):
    """Raised when self-feed terminates due to an error.

    Attributes
    ----------
    steps_survived : int
        Number of self-feed steps completed successfully before the error.
    """

    def __init__(self, steps_survived: int):
        super().__init__(f"Self-feed failed after {steps_survived} steps")
        self.steps_survived = steps_survived


def calculate_macros(
    data,
    output_dir,
    dataset_root,
    steps_to_predict,
    use_pbc=True,
    device="cpu",
    legacy_dataset=None,
):
    """Calculate and plot macros for a self-feed run.

    Parameters
    ----------
    data : dict
        Dictionary with ``target`` and ``pred`` entries containing tensors.
    output_dir : str
        Directory where macro plots/data will be written.
    dataset_root : str
        Path to the simulation dataset containing ``simulation_args.json``.
    steps_to_predict : int
        Number of macro steps predicted by the model.
    use_pbc : bool, optional
        Whether the dataset uses periodic boundary conditions, by default True.
    device : str or torch.device, optional
        Device for any heavy calculations, by default ``"cpu"``.
    legacy_dataset : bool, optional
        Force using the legacy simulation assets layout.  If ``None`` (default)
        the layout is detected automatically based on ``simulation_args.json``
        and presence of ``system.xml`` in the input folder.
    """
    sim_args = None
    if legacy_dataset is None:
        try:
            sim_args = load_simulation_args(
                os.path.join(dataset_root, "simulation_args.json")
            )
            input_folder = sim_args.get("input_folder", dataset_root)
        except FileNotFoundError:
            legacy_dataset = True

    input_folder = dataset_root if legacy_dataset else input_folder

    if sim_args is None and os.path.exists(
        os.path.join(dataset_root, "simulation_args.json")
    ):
        sim_args = load_simulation_args(
            os.path.join(dataset_root, "simulation_args.json")
        )
    if sim_args is not None:
        base_step_fs = sim_args["time_step"] * sim_args["report_step"]
    else:
        base_step_fs = 5.0
    rollout_timestep_fs = base_step_fs * steps_to_predict

    pdb_path = get_dataset_pdb_path(dataset_root)
    print(f"Using PDB path: {pdb_path} from dataset root: {dataset_root}")

    try:
        md_macros.calculate_energy_macro(
            data["target"],
            data["pred"],
            dataset_root=dataset_root,
            steps_to_predict=steps_to_predict,
            pdb_path=pdb_path,
            save_data_dir=output_dir,
            use_pbc=use_pbc,
            show=False,
            device=device,
        )
    except Exception as e:
        print(f"Error calculating energy macro: {e}")
        print(traceback.format_exc())
        print("Skipping energy macro calculation.")
    try:
        md_macros.calculate_rdf_macro(
            data["target"],
            data["pred"],
            use_pbc=use_pbc,
            show=False,
            save_data_dir=output_dir,
            pdb_path=pdb_path,
        )
    except Exception as e:
        print(f"Error calculating RDF macro: {e}")
        print("Skipping RDF macro calculation.")
    try:
        if "atomic_numbers" in data["target"] and "atomic_numbers" in data["pred"]:
            md_macros.calculate_element_rdf_macro(
                data["target"],  # your dict with "pos" & "atomic_numbers"
                data["pred"],  # same keys
                use_pbc=True,
                pdb_path=pdb_path,
                save_data_dir=output_dir,
                show=False,
            )
    except Exception as e:
        print(f"Error calculating element RDF or VDOS macro: {e}")
        print("Skipping element RDF or VDOS macro calculation.")
    try:
        md_macros.calculate_element_vdos_macro(
            data["target"],
            data["pred"],
            timestep_fs=rollout_timestep_fs,
            save_data_dir=output_dir,
            show=False,
        )
    except Exception as e:
        print(f"Error calculating element RDF or VDOS macro: {e}")
        print("Skipping element RDF or VDOS macro calculation.")
    try:
        md_macros.calculate_collisions_macro(
            data["target"], data["pred"], show=False, save_data_dir=output_dir
        )
    except Exception as e:
        print(f"Error calculating collisions macro: {e}")
        print("Skipping collisions macro calculation.")
    try:
        md_macros.calculate_sharp_acceleration_macro(
            data["target"], data["pred"], show=False, save_data_dir=output_dir
        )
    except Exception as e:
        print(f"Error calculating sharp acceleration macro: {e}")
        print("Skipping sharp acceleration macro calculation.")
    try:
        md_macros.calculate_vacf_macro(
            data["target"],
            data["pred"],
            save_data_dir=output_dir,
            show=False,
        )
    except Exception as e:
        print(f"Error calculating VACF macro: {e}")
        print("Skipping VACF macro calculation.")
    try:
        md_macros.calculate_ethane_angle_switch_macro(
            data["target"],
            data["pred"],
            pdb_path=pdb_path,
            show=False,
            save_data_dir=output_dir,
        )
    except Exception as e:
        print(f"Error calculating ethane angle switch macro: {e}")
        print("Skipping ethane angle switch macro calculation.")


def denormalize(dataloader, pred, name):
    if hasattr(dataloader.dataset, "scalers"):
        if name not in dataloader.dataset.scalers:
            return pred
        scaled = dataloader.dataset.scalers[name].inverse_transform(pred.cpu())
        return torch.from_numpy(scaled).to(pred.device)
    return pred


def build_sample_pbc(data, sample, cutoff_radius, dtype=torch.float64):
    box_lengths_nm = sample.box_vectors.cpu().numpy()
    wrapped_positions, box_size = wrap_positions(
        data["pred"]["pos"][-1], box_lengths_nm=box_lengths_nm
    )
    edge_index, edge_distances = build_graph_features_pbc(
        wrapped_positions, box_size, cutoff_radius
    )

    edge_index = torch.tensor(np.ascontiguousarray(edge_index), dtype=torch.int64)
    edge_distances = torch.tensor(np.ascontiguousarray(edge_distances), dtype=dtype)

    pred_pos = data["pred"]["pos"][-1]
    pred_vel = data["pred"].get("vel", [torch.zeros_like(sample.vel)])[-1]
    pred_force = data["pred"].get("force", [torch.zeros_like(sample.force)])[-1]

    return Data(
        pos=pred_pos,
        vel=pred_vel,
        force=pred_force,
        mass=sample.mass,
        charge=sample.charge,
        edge_index=edge_index,
        rel_pos=edge_distances,
        box_vectors=sample.box_vectors,
    )


def build_sample(data, sample, device="cpu"):
    edge_index = build_graph_with_knn(
        data["pred"]["pos"][-1].to(device),
        1,
        len(data["pred"]["pos"][-1]),
        device=device,
        num_neighbors=len(data["pred"]["pos"][-1]) - 1,
    )
    pred_pos = data["pred"]["pos"][-1]
    pred_vel = data["pred"].get("vel", [torch.zeros_like(sample.vel)])[-1]
    pred_force = data["pred"].get("force", [torch.zeros_like(sample.force)])[-1]

    return Data(
        pos=pred_pos,
        vel=pred_vel,
        force=pred_force,
        mass=sample.mass,
        charge=sample.charge,
        edge_index=edge_index,
    )


def remove_drift(data):
    if "vel" in data["pred"]:
        data["pred"]["vel"][-1] -= data["pred"]["vel"][-1].mean(axis=0)
    return data


@torch.no_grad()
def self_feed(dataloader, model, base_data_dir, args, limit_steps=None, device="cpu"):

    self_inferred_sample = None
    _sample = next(iter(dataloader.dataset))[0]

    data = {
        "input": {},
        "pred": {},
        "target": {},
        "mass": _sample.mass,
        "charge": _sample.charge,
        "force": _sample.force,
    }

    progress_bar = tqdm(total=None)  # No total, dynamically updated
    dataset_iter = iter(dataloader.dataset)
    data_left = True

    i = 0
    last_returned = -1
    num_restarts = 0
    tries = 0
    MAX_TRIES = 20

    while True:
        try:
            if data_left and i > last_returned:
                try:
                    for _ in range(args.steps_to_predict):
                        sample = next(dataset_iter)
                except StopIteration:
                    data_left = False
            # use sample in the first step
            input_sample = self_inferred_sample or sample[0].to(device)
            model_input = dataloader.transform(input_sample)
            pred = model(model_input).detach().cpu()
        except KeyboardInterrupt:
            print("Interrupting self feed")
            break
        except Exception as e:
            print(f"Errored at step {i}, exiting self-feed loop")
            raise SelfFeedError(i) from e

        # calculate and assign the next step
        offset = 0
        for key in ["pos", "vel", "force"]:
            if data_left:
                data["input"].setdefault(key, []).append(getattr(sample[0], key))

            if key not in args.target and f"{key}_dt" not in args.target:
                continue

            data["pred"].setdefault(key, [])
            prediction_current = pred[:, offset : offset + 3]
            prediction_current = denormalize(dataloader, prediction_current, key)

            if data_left:
                target_current = sample[0].y[:, offset : offset + 3]
                data["target"].setdefault(key, [])

            if f"{key}_dt" in args.target:

                data["pred"][key].append(
                    input_sample.__getattribute__(key).cpu() + prediction_current.cpu()
                )
                if data_left:
                    data["target"][key].append(data["input"][key][-1] + target_current)
            else:
                data["pred"][key].append(prediction_current.cpu())
                if data_left:
                    data["target"][key].append(target_current)
            offset += 3

        # handle data explosions
        if (pred > 1e9).any():
            print(f"=== Prediction exploded at step {i} ===")

            if tries == 0:
                num_restarts += 1
            if i <= last_returned:
                tries += 1
                print(
                    f"Last time returned from {last_returned}, this time would return from {i}. Try {tries}/{MAX_TRIES}"
                )
                if tries == MAX_TRIES:
                    print("Maximum number of tries reached, exiting self-feed")

                    break
            else:
                tries = 0
            if tries == 0:
                last_returned = i

            steps_to_return = STEPS_TO_RETURN_MULTIPLIER * (tries + 1)
            if i <= steps_to_return:
                print(f"Won't return {steps_to_return} from step {i}, duh")
                break
            print(f"Returning from step {i} to {i - steps_to_return}")

            i = i - steps_to_return
            # remove data we will generate again
            for data_type, data_type_value in data.items():
                if data_type in ["pred"]:
                    for data_unit, _data_unit_value in data_type_value.items():
                        data[data_type][data_unit] = data[data_type][data_unit][:i]
        try:
            data = remove_drift(data)
            dtype = (
                torch.float64
                if args.precision_mode == PrecisionMode.DOUBLE
                else torch.float32
            )
            if args.use_pbc is True:
                # Ensure box vectors are present; if missing, use processed.pdb
                if (
                    not hasattr(sample[0], "box_vectors")
                    or sample[0].box_vectors is None
                ):
                    pdb_path = get_dataset_pdb_path(base_data_dir)
                    dims_ang = get_box_dimensions(pdb_path)
                    box_lengths_nm = (dims_ang[:3] / 10.0).astype("float32")
                    sample[0].box_vectors = torch.tensor(
                        box_lengths_nm, dtype=dtype, device=sample[0].pos.device
                    )
                self_inferred_sample = build_sample_pbc(
                    data, sample[0], args.cutoff_radius, dtype=dtype
                ).to(device)
            else:
                self_inferred_sample = build_sample(data, sample[0], dtype).to(device)

            if limit_steps is not None and i == limit_steps:
                break

            i += 1
            progress_bar.n = i
            progress_bar.refresh()
        except KeyboardInterrupt:
            print("Interrupting self feed")
            break
        except Exception as e:
            print(f"Errored at step {i}, exiting self-feed loop")
            raise SelfFeedError(i) from e
    progress_bar.close()
    print(
        f"{num_restarts} restarts were needed. Average number of steps survived: {i / (num_restarts + 1):.2f}"
    )
    # if we restarted, targets and inputs are potentially longer - we will trim to their length
    for data_type, data_type_value in data.items():
        if data_type in ["input", "target"]:
            for data_unit, _data_unit_value in data_type_value.items():
                if data_unit in data["pred"]:
                    data[data_type][data_unit] = data[data_type][data_unit][
                        : len(data["pred"][data_unit])
                    ]

    return data


def main():
    args, _ = parse_args()
    model = create_model(args)

    dataloader = load_class_from_args(args, "dataloader")
    train_dataloader = dataloader(args, partition="train")
    validation_dataloader = dataloader(args, partition="valid")
    trainer = load_class_from_args(args, "trainer")
    trainer = trainer(
        model, train_dataloader, validation_dataloader=validation_dataloader, args=args
    )

    base_data_dir = validation_dataloader.dataset.base_data_dir

    # run self feed on validation dataset
    data = self_feed(
        validation_dataloader,
        trainer.model,
        base_data_dir,
        args,
        limit_steps=2000,
        device=trainer.device,
    )

    # save output data to run directory
    os.makedirs(trainer.save_dir_path, exist_ok=True)
    with open(os.path.join(trainer.save_dir_path, "self_feed_data.pkl"), "wb") as file:
        pickle.dump(data, file)

    pdb_output_path = os.path.join(trainer.save_dir_path, "processed.pdb")
    sim_args = load_simulation_args(os.path.join(base_data_dir, "simulation_args.json"))
    os.symlink(
        get_dataset_pdb_path(base_data_dir),
        pdb_output_path,
    )

    print("Generating macros...")
    calculate_macros(
        data,
        os.path.join(trainer.save_dir_path, MACROS_DIR_NAME),
        base_data_dir,
        args.steps_to_predict,
        args.use_pbc,
        device=trainer.device,
    )

    print(f"Saved macros to {trainer.save_dir_path}")


if __name__ == "__main__":
    main()
