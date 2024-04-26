# pySSEM - Source Sink Evolutionary Model

**This is still at pre-alpha stage, the model is still actively being developed and tested. Please do not rely on results.**

## Description

pySSEM is a tool that investigates the evolution of the space objects population in Low Earth Orbit (LEO) by exploiting a new probabilistic source-sink model. The objective is to estimate the LEO orbital capacity. This is carried out through the long-term propagation of the proposed source-sink model, which globally takes into account different object species, such as active satellites, derelict satellites, debris, and additional subgroups. Since the Space Objects (SOs) are propagated as species, the information about single objects is missing, but it allows the model to be computationally fast and provide essential information about the projected future distribution of SOs in the space environment for long prediction horizons.

## Installation

Currently, to install pySSEM, you just need to pull the git repository and run `pyssem.py`.

```bash
git clone <repository_url>
cd <repository_directory>
python pyssem.py# pyssem
Python port of MOCAT-SSEM framework
