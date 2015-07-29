#include <mlpack/core.hpp>
#include <mlpack/methods/pca/pca.hpp>

// Convenience.
using namespace mlpack;

int main()
{
	// First, load the data.
	arma::mat data;
//	arma::mat out_data;
//	arma::vec eigval;
//	arma::mat eigvec;
	pca::PCA mypca(false);
	double ratio;

	data::Load("./small_data/haberman.csv", data, true);
	ratio = mypca.Apply(data, 0.9);

	// Save the output.
	std::cout << "Real ratio is: " << ratio << std::endl;
	data::Save("./small_data/pca.csv", data, true);
//	data::Save("./small_data/eval.csv", eigval, true);
//	data::Save("./small_data/evec.csv", eigvec, true);
}
