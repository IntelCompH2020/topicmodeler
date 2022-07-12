# Dockerfile explanation
A dockerfile is a file containing all the instructions to generate a docker image, a template to build a docker container which contains the application.

- Base image: the base image for the application is ubuntu+python
- Directories architecture: the application will be in the */app* directory inside the container, and the */data* directory will contain shared data with host.
- Application: copy `manageLists.py` to current directory (*/app*)
- Execution: when the container is created, the command in **ENTRYPOINT** will be executed. The command in **CMD** is the default option when executing the container and can be overwritten by the user.


# Create Docker image with WordList manager
This is the required command to create a docker image with the the word-list manager. Both files, `Dockerfile` and `manageLists.py` are required.

The flag *-t* specifies the name of the image that will be created with an optional tag (for example its version).
```
docker build <-t NAME:tag> <Dockerfile location>
```
## Example:
```
docker build -t mng-lsts .
```
- The name of the image will be *mng-lsts*, with no specific version.
- The location of the `Dockerfile` is the current directory.

# Run image container
To run a container with the previous configuration, the following command is needed:
```
docker run <--rm> -i <--name CONTAINER-NAME> -v path/to/local/wordlists:/data/wordlists IMAGE-NAME
```
Flags:
- --rm: remove the container when execution ends. (optional)
- -i: set interactive mode. It is required to use standard input
- --name: a name for the container. (optional)
- v: volume binding. It maps a local directory to a directory inside the container so that local files can be accessed from it. The format is:
`/absolute/path/to/local/dir`:`/absolute/path/to/container/dir`


## Examples
### Execute help
Help is the default option when executing the container, as specified in `Dockerfile`
```
docker run --rm -i --name cnt-ja-ml -v /Users/joseantem/Documents/github/topicmodeler/wordlists:/data/wordlists mng-lsts
```

### List of wordlists
```
docker run --rm -i --name cnt-ja-ml -v /Users/joseantem/Documents/github/topicmodeler/wordlists:/data/wordlists mng-lsts --listWordLists
```

### Create a new word list
```
docker run --rm -i --name cnt-ja-ml -v /Users/joseantem/Documents/github/topicmodeler/wordlists:/data/wordlists mng-lsts --createWordList
```
Once this command is executed, the container will read the input in terminal until end of file command is pressed. You can copy and paste in the terminal the following example of a list, then press **CTRL+D** (*Note*: if on Windows, press **CTRL+Z** and then press **enter**.):
```
{
  "name": "test",
  "description": "This is just a test to check everything works fine.",
  "valid_for": "stopwords",
  "visibility": "Public",
  "wordlist": [
    "w1",
    "w2",
    "w3"
  ]
}

```

### Copy wordlist
```
docker run --rm -i --name cnt-ja-ml -v /Users/joseantem/Documents/github/topicmodeler/wordlists:/data/wordlists mng-lsts --copyWordList test
```

### Rename wordlist
```
docker run --rm -i --name cnt-ja-ml -v /Users/joseantem/Documents/github/topicmodeler/wordlists:/data/wordlists mng-lsts --renameWordList test-copy test_backup
```


### Delete wordlist
```
docker run --rm -i --name cnt-ja-ml -v /Users/joseantem/Documents/github/topicmodeler/wordlists:/data/wordlists mng-lsts --deleteWordList test
```
