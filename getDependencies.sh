EIGEN='Eigen.tar.gz'
SOURCE='eigen-eigen-b30b87236a1b/Eigen'

echo Now cloning the Mavlink library
git clone https://github.com/mavlink/c_library.git

echo Now downloading Eigen
wget http://bitbucket.org/eigen/eigen/get/3.2.7.tar.gz -O ${EIGEN}

echo Download done

echo Now unpacking to Eigen.
tar -zxvf ${EIGEN} ${SOURCE} --strip 1
echo ad moving to eigen_library
mv Eigen eigen_library

echo Unpacking done
echo Removing archive ${EIGEN}

rm ${EIGEN}

