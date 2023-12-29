# -*- coding: utf-8 -*-

"""
Swarm Operation Backend

This module abstracts various operations in the swarm such as updating
the personal best, finding neighbors, etc. You can use these methods
to specify how the swarm will behave.
"""

# Import standard library
import logging

# Import modules
import numpy as np

from ..utils.reporter import Reporter
from .handlers import BoundaryHandler, VelocityHandler
from functools import partial


rep = Reporter(logger=logging.getLogger(__name__))


def compute_pbest(swarm): #應該考慮擁有相同層數的 drf，但此處是各個粒子的 best 無關層數。
    """Update the personal best score of a swarm instance

    You can use this method to update your personal best positions.

    .. code-block:: python

        import pyswarms.backend as P
        from pyswarms.backend.swarms import Swarm

        my_swarm = P.create_swarm(n_particles, dimensions)

        # Inside the for-loop...
        for i in range(iters):
            # It updates the swarm internally
            my_swarm.pbest_pos, my_swarm.pbest_cost = P.update_pbest(my_swarm)

    It updates your :code:`current_pbest` with the personal bests acquired by
    comparing the (1) cost of the current positions and the (2) personal
    bests your swarm has attained.

    If the cost of the current position is less than the cost of the personal
    best, then the current position replaces the previous personal best
    position.

    Parameters
    ----------
    swarm : pyswarms.backend.swarm.Swarm
        a Swarm instance

    Returns
    -------
    numpy.ndarray
        New personal best positions of shape :code:`(n_particles, n_dimensions)`
    numpy.ndarray
        New personal best costs of shape :code:`(n_particles,)`
    """
    try:
        # Infer dimensions from positions
        dimensions = swarm.dimensions
        # Create a 1-D and 2-D mask based from comparisons
        #print("swarm.current_cost: ", swarm.current_cost)
        #print("swarm.pbest_cost: ", swarm.pbest_cost)
        mask_cost = swarm.current_cost < swarm.pbest_cost
        mask_pos = np.repeat(mask_cost[:, np.newaxis], dimensions, axis=1)
        
        
        #print("swarm.position: ", swarm.position)
        #print("swarm.pbest_pos: ", swarm.pbest_pos)
        
        # Apply masks
        new_pbest_pos = np.where(~mask_pos, swarm.pbest_pos, swarm.position)
        new_pbest_cost = np.where(
            ~mask_cost, swarm.pbest_cost, swarm.current_cost
        )
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return (new_pbest_pos, new_pbest_cost)


def compute_velocity(swarm, clamp, vh, bounds=None):
    """Update the velocity matrix

    This method updates the velocity matrix using the best and current
    positions of the swarm. The velocity matrix is computed using the
    cognitive and social terms of the swarm. The velocity is handled
    by a :code:`VelocityHandler`.

    A sample usage can be seen with the following:

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm, VelocityHandler

        my_swarm = P.create_swarm(n_particles, dimensions)
        my_vh = VelocityHandler(strategy="invert")

        for i in range(iters):
            # Inside the for-loop
            my_swarm.velocity = compute_velocity(my_swarm, clamp, my_vh, bounds)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    clamp : tuple of floats, optional
        a tuple of size 2 where the first entry is the minimum velocity
        and the second entry is the maximum velocity. It
        sets the limits for velocity clamping.
    vh : pyswarms.backend.handlers.VelocityHandler
        a VelocityHandler object with a specified handling strategy.
        For further information see :mod:`pyswarms.backend.handlers`.
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.

    Returns
    -------
    numpy.ndarray
        Updated velocity matrix
    """
    try:
        # Prepare parameters
        swarm_size = swarm.position.shape
        c1 = swarm.options["c1"]
        c2 = swarm.options["c2"]
        w = swarm.options["w"]
        # Compute for cognitive and social terms
        cognitive = (
            c1
            * np.random.uniform(0, 1, swarm_size)
            * (swarm.pbest_pos - swarm.position)
        )
        
        #if round(swarm.best_pos[0]) < round(best_position[0]):
            #不參考 best position 沒有用到的參數，即用原本的
            
        #if round(best_position[0]) > round(best_position[0]):
            #不參考 position 沒有用到的參數，即用 best position 的
            
        best_minus_pos = np.empty(shape=[0, swarm.position.shape[1]])
        #print("best_minus_pos: ", best_minus_pos)
        print("\nParticles num:", swarm.position.shape[0]) #共有幾個粒子
        for i in range(swarm.position.shape[0]):
            if round(swarm.position[i, 0]) > round(swarm.best_pos[0]): #best drf 層數沒有當層多
                best_pos_drf = np.append(np.array([swarm.best_pos[i] for i in range(swarm.best_pos.size) if i < round(swarm.best_pos[0])+1]), swarm.position[i, int(round(swarm.best_pos[0])+1):])
                best_minus_pos = np.append(best_minus_pos, [best_pos_drf - swarm.position[i]], axis = 0)
            else:
                best_minus_pos = np.append(best_minus_pos, [swarm.best_pos - swarm.position[i]], axis = 0)
            
        print("social velocity:\n ", best_minus_pos)
        #print("swarm size: ", swarm_size)
        
        social = ( #此處更新應以有相同層數的 drf 為主
            c2
            * np.random.uniform(0, 1, swarm_size)
            * (best_minus_pos)
        )
        # Compute temp velocity (subject to clamping if possible)
        temp_velocity = (w * swarm.velocity) + cognitive + social
        updated_velocity = vh(
            temp_velocity, clamp, position=swarm.position, bounds=bounds
        )

    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    except KeyError:
        rep.logger.exception("Missing keyword in swarm.options")
        raise
    else:
        return updated_velocity


def compute_position(swarm, bounds, bh):
    """Update the position matrix

    This method updates the position matrix given the current position and the
    velocity. If bounded, the positions are handled by a
    :code:`BoundaryHandler` instance

    .. code-block :: python

        import pyswarms.backend as P
        from pyswarms.swarms.backend import Swarm, VelocityHandler

        my_swarm = P.create_swarm(n_particles, dimensions)
        my_bh = BoundaryHandler(strategy="intermediate")

        for i in range(iters):
            # Inside the for-loop
            my_swarm.position = compute_position(my_swarm, bounds, my_bh)

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    bounds : tuple of numpy.ndarray or list, optional
        a tuple of size 2 where the first entry is the minimum bound while
        the second entry is the maximum bound. Each array must be of shape
        :code:`(dimensions,)`.
    bh : pyswarms.backend.handlers.BoundaryHandler
        a BoundaryHandler object with a specified handling strategy
        For further information see :mod:`pyswarms.backend.handlers`.

    Returns
    -------
    numpy.ndarray
        New position-matrix
    """
    try:
        temp_position = swarm.position.copy()
        temp_position += swarm.velocity

        if bounds is not None:
            temp_position = bh(temp_position, bounds)

        position = temp_position
    except AttributeError:
        rep.logger.exception(
            "Please pass a Swarm class. You passed {}".format(type(swarm))
        )
        raise
    else:
        return position


def compute_objective_function(swarm, objective_func, pool=None, **kwargs):
    """Evaluate particles using the objective function

    This method evaluates each particle in the swarm according to the objective
    function passed.

    If a pool is passed, then the evaluation of the particles is done in
    parallel using multiple processes.

    Parameters
    ----------
    swarm : pyswarms.backend.swarms.Swarm
        a Swarm instance
    objective_func : function
        objective function to be evaluated
    pool: multiprocessing.Pool
        multiprocessing.Pool to be used for parallel particle evaluation
    kwargs : dict
        arguments for the objective function

    Returns
    -------
    numpy.ndarray
        Cost-matrix for the given swarm
    """
    if pool is None:
        return objective_func(swarm.position, **kwargs)
    else:
        results = pool.map(
            partial(objective_func, **kwargs),
            np.array_split(swarm.position, pool._processes),
        )
        return np.concatenate(results)

# 設第一個參數為層數，第二、三個參數為各層顆樹
#1st
#[1,2,3] 0.3 3其實是用不到的，但會因為效果比較好，去影響到 [2,4,5] 的 5
#[2,4,5] 0.4

#需要在更新 social 時讓第二個粒子知道不該參考第三個值。

#2nd
#pbest
#[1,2,3] 0.5
#[2,4,5] 0.4
#current
#[1,3,4] 0.6
#[2,5,6] 0.5

#3rd
#pbest
#[1,2,3] 0.5
#[2,4,5] 0.4
#current
