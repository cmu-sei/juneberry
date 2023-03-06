#! /usr/bin/env python3

# ======================================================================================================================
# Juneberry - Release 0.5
#
# Copyright 2022 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS"
# BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER
# INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED
# FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM
# FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Released under a BSD (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution. Please see
# Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM22-0856
#
# ======================================================================================================================

import logging
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

"""
This file contains support for working with cuda devices AND distributed (multiprocessing) mode. 
PyTorch is very flexible and allows a lot of options and we don't use them all.  This details
the use cases and how we approach it.

# TL;DR

We focus on a few specific cases.

Case 1: CPU only
Case 2: One GPU using non-distributed support
Case 3: Multi-GPU distributed

We currently do NOT support multi host options.

# DETAILS

Summary of PyTorch terms:
node - Compute host.
num_gpus - How many GPUs on this node?
gpu - The gpu number of a process on a node.
world_size - How many total compute processes in the "world" (all compute hosts together sharing effort).
rank - Process number within the world. Rank 0 is the coordinator.

Invariants that hold true for pytorch in general:
  gpu < num_gpus
  num_gpus <= world_size
  rank < world_size

Implications of not supporting multiple nodes:
  world_size == num_gpus
  rank == gpu

# Distributed mode

In distributed mode, we use the pytorch multi-processing infrastructure. PyTorch's multiprocessing
supports processes across multiple nodes. We do NOT use this because we only support single node
use cases. Because of this, when num_gpus is 1, we do NOT need distributed mode. We do not currently
allow the user to specify if they want distributed mode or not.

# Interaction with torch.cuda

When number of GPUs not specified:

In the case where the number of GPUs is NOT specified, we use torch.cuda to determine availability
and the number of GPUs. If GPUs are available we use them all (constrained externally by 
CUDA_VISIBLE_DEVICES) and select distributed mode if more than one gpu is available. If not 
available we fall back to CPU mode. 

When number of GPUs is specified:

When we are provided with a number of desired GPUs we check to see if CUDA is available and if
the number of GPUs is available, constrained externally by CUDA_VISIBLE_DEVICES. If GPUs
are requested and CUDA is not available or the number of GPUs are not available, we report 
an error and exit. We assume that if the number of GPUs is specified then the user really 
wants that, and it isn't useful continuing if we can't meet that need.

"""


def determine_gpus(num_gpus: int = None):
    """
    If None is specified for num_gpus, determines how many are available.
    If num_gpus is specified, check to see if that number is available, exiting if not.
    :param num_gpus: The number of gpus desired, or None to auto-detect.
    :return: The number of gpus to use.
    """
    if num_gpus is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
        else:
            num_gpus = 0
        logger.info(f"Detected {num_gpus} gpus. CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES')}'")
    elif num_gpus > 0:
        if not torch.cuda.is_available():
            logger.error(f"{num_gpus} requested, but CUDA not available. EXITING.")
            sys.exit(-1)

        if num_gpus > torch.cuda.device_count():
            logger.error(f"{num_gpus} requested, but only {torch.cuda.device_count()} available. EXITING.")
            sys.exit(-1)

    return num_gpus


def setup_cuda_device(num_gpus, gpu):
    """
    Sets up the cuda device, initializing the distributed environment if necessary.
    :param num_gpus: The total number of gpus.
    :param gpu: This current gpu.
    :return: The device
    """
    if num_gpus == 0:
        device = torch.device("cpu")
    else:
        assert gpu < num_gpus
        device = torch.device("cuda:" + str(gpu))
        torch.cuda.set_device(device)
        # Adding LOCAL_RANK env to support pytorch 22.11 upgrade
        os.environ['LOCAL_RANK'] = str(device)

    return device


def prepare_for_distributed() -> None:
    """
    Used by the distribution process to set common values needed for distribution.
    :return: None
    """

    import socket
    import errno

    # Attempt to fetch min and max ports for Juneberry if they exist as environment variables, otherwise
    # use the default port for the min and (100 + min) for the max.
    port = int(os.environ['JUNEBERRY_MIN_PORT']) if "JUNEBERRY_MIN_PORT" in os.environ else 63557
    max_port = int(os.environ['JUNEBERRY_MAX_PORT']) if "JUNEBERRY_MAX_PORT" in os.environ else port + 100

    # Set up a socket.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    logger.info(f"Preparing to initialize the distributed process group...")
    logger.info(f"Checking available ports between {port} and {max_port}")

    # Test ADDR:PORT combinations until an available connection is found. The first available
    # combination will be used later to initialize the distributed process group.
    while port <= max_port:
        try:
            logger.info(f"Checking if localhost:{port} is available.")
            s.bind(('localhost', port))
            break

        except socket.error as e:

            # If the address is in use, increment the port and try again.
            if e.errno == errno.EADDRINUSE:
                logger.warning(f"localhost:{port} appears to be in use. Incrementing the port "
                               f"and trying again.")
                port += 1

            # If some other error is encountered, print the error and exit.
            else:
                logger.error(f"Unexpected socket error. Exiting.")
                logger.error(f"{e}")
                sys.exit(-1)

    logger.info(f"localhost:{port} appears to be available.")

    # Some environment variables for the process group
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)


def setup_distributed(world_size, rank) -> None:
    """
    Called by spawned processes to work together in a distributed world.
    :param world_size: The number of nodes in the world.
    :param rank: The rank of this node.
    :return: None
    """
    # Join the process group
    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            world_size=world_size,
                            rank=rank)


def prepare_model(distributed, num_gpus, gpu, model, device):
    """
    Prepares the model according to the to the number of gpus on the host.
    :param distributed: Are we in distributed mode?
    :param num_gpus: The total number of gpus on this host.
    :param gpu: The gpu on this host.
    :param model: The model to prepare.
    :param device: The device (if any) to use.
    :return: A model prepared correctly for the desired hardware.
    """
    if distributed:
        # In distributed mode, we send the model to the device.
        logger.info(f"Setting up model for device={device}")
        model.to(device)
        return torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    elif num_gpus == 1:
        logger.info(f"Setting up model for device={device}")
        model.to(device)
        return torch.nn.DataParallel(model)

    # Default case is CPU so no changes to the model.
    return model


def log_cuda_configuration(num_gpus, gpu, the_logger=logger):
    """
    Writes the specified settings to the provided logger.
    :param num_gpus: The number of gpus in total.
    :param gpu: Which gpu this is?
    :param the_logger: The logger.
    :return:
    """
    if num_gpus == 0:
        the_logger.info("Compute Mode: CPU")
    elif num_gpus == 1:
        the_logger.info("Compute Mode: Single GPU; NOT DISTRIBUTED")
    else:
        the_logger.info(f"Compute Mode: GPU: gpu(rank)={gpu}, num_gpus(world_size)={num_gpus}")


def start_distributed(function, num_gpus: int) -> None:
    # TODO: Add logging cleanup/prep here

    # Spawn one training process per GPU.
    logger.info(f"*** Spawning {num_gpus} processes...")
    mp.spawn(function, nprocs=num_gpus, join=True)
