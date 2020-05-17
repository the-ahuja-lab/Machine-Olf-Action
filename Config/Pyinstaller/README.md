1) Open command prompt and navigate to directory project directory
2) Make sure you have the appropriate conda environment with all required libraries and then activate conda environment
3) Change paths inside (added_files, pathex variables) run.spec to the correct respecctive locations of the project
3) Again make sure you are in the directory where all project files are there and run.spec file is present
3) Run command "pyinstaller --onefile run.spec" (without quotes) to build exe
4) Once step3 is completed successfully eithout errors, exe will be present in the newly created dist directory

