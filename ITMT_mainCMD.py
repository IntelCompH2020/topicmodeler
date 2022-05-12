"""
*** IntelComp H2020 project ***
*** Interactive Topic Model Trainer ***

Application for the interactive training of Topic Models

It implements a command line tool for the training and curation of topic models
exploiting the tools available in topicmodeling.py:
   - Different LDA implementations
   - Topic assessment tools
   - Topic curation tools

Contributors: José Antonio Espinosa-Melchor
              Lorena Calvo-Bartolomé
              Jerónimo Arenas-García (jarenas@ing.uc3m.es)

Menú navigator is based on the base class created by Jesús Cid-Sueiro
(https://github.com/Orieus/menuNavigator)
"""

import os
import pathlib
import argparse

#Local imports
from src.menu_navigator.menu_navigator import MenuNavigator
from src.ITMTmanager import TaskManager

# ########################
# Main body of application
# ########################
def main():
    # ####################
    # Read input arguments

    # settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', type=str, default=None,
                        help="path to a new or an existing project")
    parser.add_argument('--f', action='store_true', default=False,
                        help='Force creation of new project. Overwrite existing.')
    args = parser.parse_args()

    if os.path.isdir(project_path):
        if not args.f:
            print('Loading the selected project')
            option = 'load'
        else:
            print('Forcing creation of a blank project')
            option = 'create'
    else:
        print('A new project will be created')
        option = 'create'
    active_options = None

    # Create TaskManager for this project
    tm = TaskManager(project_path)

    # ########################
    # Prepare user interaction
    # ########################
    paths2data = {}
    path2menu = pathlib.Path('config', 'ITMTmenu.yaml')

    # ##############
    # Call navigator
    # ##############

    menu = MenuNavigator(tm, path2menu, paths2data)
    menu.front_page(title="Interactive Topic Model Trainer for the IntelComp H2020 project")
    menu.navigate(option, active_options)


# ############
# Execute main
if __name__ == '__main__':
    main()
