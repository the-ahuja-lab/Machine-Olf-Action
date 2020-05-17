# Machine-OLF-action: a machine learning-driven computational framework

Description of all the folders present in the repository

### Code - contains source code of the application
To run application using the source code
1. Clone the git repository
1. All the source code is present in the "Code" folder.
1. Download cond/pip packages required by code using "environment.yml" file present in the Config/CondaEnvironment/ folder.
1. Make sure you have conda available on your system. If not download conda first and then create a new conda environment using command "conda env create -f environment.yml"
1. You can use your favourite IDE (like PyCharm) with the above newly created conda environment to run code.
1. Run "run.py" file to start the flask application server which then opens the application in your default browser.

### Config - contains confiuration files needed to run and buid the executables of application
**environment.yml** - Conda environment file with list of packages to replicate environment on your machine  
**Pyinstaller** - Details on how to create executable files of different platforms from the source code

### Documents - documentation of the application
Documentation related to the application

### Executables - executables of the application
To directly run the application, go to Executables folder and download executable for the respective platform
