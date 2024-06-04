DOCKER_FILE="Dockerfile"

if [ -z $USER ];
then
    USER=$(whoami)
fi

if [ -z $UID ];
then
    UID=$(id -u)
fi

if [ -z $GID ];
then
    GID=$(id -g)
fi

echo "Passing arguments to docker build: USER: $USER, UID: $UID, GID: $GID"

docker build --build-arg USER=$USER --build-arg UID=$UID --build-arg GID=$GID -f $DOCKER_FILE -t sofia_fairml4h --no-cache .