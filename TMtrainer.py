"""
*** IntelComp H2020 project ***
*** Topic Model Trainer App ***

Application for the training of Topic Models using the Topic Modeling Toolbox

It exemplifies the use of the classes in topicmodeling.py to provide the functionality
that will be needed in the Interactive Model Trainer

The application structure is derived from: https://github.com/Orieus/menuNavigator
"""

import os
import argparse
from TMnavigator.menu_navigator import MenuNavigator
from TMnavigator.TMmanager import TaskManager
import time

# ####################
# Read input arguments

# settings
parser = argparse.ArgumentParser()
parser.add_argument('--p', type=str, default=None,
                    help="path to a new or an existing project")
parser.add_argument('--f', action='store_true', default=False,
                    help='Overwrite existing project')
args = parser.parse_args()

# Read project_path
project_path = args.p
if args.p is None:
    while project_path is None or project_path == "":
        project_path = input('-- Write the path to the project to load or '
                             'create: ')
else:
    project_path = args.p

if os.path.isdir(project_path):
    if not args.f:
        print('Loading the selected project')
        option = 'load'
    else:
        print('Forcing creation of a blank project')
        option = 'create'
else:
    print('The project will be created')
    option = 'create'
active_options = None

# Create TaskManager for this project
tm = TaskManager(project_path)

# ########################
# Prepare user interaction
# ########################
paths2data = {}
path2menu = './TMnavigator/TMmenu.yaml'

# ##############
# Call navigator
# ##############
menu = MenuNavigator(tm, path2menu, paths2data)
menu.front_page(title="Interactive Topic Model Trainer for the IntelComp H2020 project")
menu.navigate(option, active_options)
