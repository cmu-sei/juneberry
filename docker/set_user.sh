#! /usr/bin/env bash

# This convenience script is designed to be executed INSIDE the juneberry contaienr
# to create a user and group with IDs that work well with the mounted volumes. This script
# requires that various user information is passed into the container. This user information
# is:
# - user name as USER_NAME
# - user id as USER_ID
# - user group id as USER_GID
#
# For example, using these options to docker run:
#
# e USER_NAME=${USER} -e USER_ID=`id -u ${USER}` -e USER_GID=`id -g ${USER}`
#
# Or via this line in enter_juneberry_container:
#
# ENVS_USER="-e USER_NAME=${USER} -e USER_ID=`id -u ${USER}` -e USER_GID=`id -g ${USER}`"
#
# As this is all temporary, this script needs to be executed on ever new container instantiation.

# Add the group and user if all three of these set
if test -n "${USER_NAME}" && test -n "${USER_ID}" && test -n "${USER_GID}"; then
	groupadd -g ${USER_GID} domain_users
	useradd -m -s /bin/bash -u ${USER_ID} -g ${USER_GID} -G root ${USER_NAME}

	# Add the default path to the bashrc. In the case of the nvidia containers they have
	# manually set the path via docker file not in a global bashrc or profile,
	# so we have no way to source that in our bashrc.
	echo "export PATH=${PATH}" >> /home/${USER_NAME}/.bashrc

	# Give ourself sudo access so we mimic a normal system where the user has sudo access
	echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

	# Now, set us to that user
	su ${USER_NAME}
else
	echo "Not setting user because USER_NAME, USER_ID or USER_GID not set."
fi
