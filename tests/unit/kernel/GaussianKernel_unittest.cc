/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/lib/config.h>

#ifdef HAVE_CXX11

#include <shogun/base/some.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/features/DenseFeatures.h>
#include <gtest/gtest.h>
#include <algorithm>

using namespace shogun;
using std::iota;
using std::for_each;

TEST(GaussianKernel, compute)
{
	const index_t dim=1;
	const index_t N=4;
	const index_t M=6;
	const float64_t width=2*pow(2, 0.05);

	SGMatrix<float64_t> data_1(dim, N);
	SGMatrix<float64_t> data_2(dim, M);

	iota(data_1.data(), data_1.data()+data_1.size(), 1);
	iota(data_2.data(), data_2.data()+data_2.size(), data_1.size()+1);

	for_each(data_1.data(), data_1.data()+data_1.size(), [&data_1](float64_t& val) { val=val/data_1.size(); });
	for_each(data_2.data(), data_2.data()+data_2.size(), [&data_2](float64_t& val) { val=val/data_2.size(); });

	auto feats_1=some<CDenseFeatures<float64_t> >(data_1);
	auto feats_2=some<CDenseFeatures<float64_t> >(data_2);

	auto kernel=some<CGaussianKernel>(10, width);
	kernel->init(feats_1, feats_2);

	auto euclidean_distance=some<CEuclideanDistance>();
	euclidean_distance->init(feats_1, feats_2);

	for (auto i=0; i<N; ++i)
	{
		for (auto j=0; j<M; ++j)
		{
			auto distance=euclidean_distance->distance(i, j);
			EXPECT_NEAR(kernel->kernel(i, j), CMath::exp(-distance*distance/width), 1E-6);
		}
	}
}

TEST(GaussianKernel, precomputed_distance)
{
	const index_t dim=1;
	const index_t N=4;
	const index_t M=6;
	const float64_t width=0.05;

	SGMatrix<float64_t> data_1(dim, N);
	SGMatrix<float64_t> data_2(dim, M);

	iota(data_1.data(), data_1.data()+data_1.size(), 1);
	iota(data_2.data(), data_2.data()+data_2.size(), data_1.size()+1);

	for_each(data_1.data(), data_1.data()+data_1.size(), [&data_1](float64_t& val) { val=val/data_1.size(); });
	for_each(data_2.data(), data_2.data()+data_2.size(), [&data_2](float64_t& val) { val=val/data_2.size(); });

	auto feats_1=some<CDenseFeatures<float64_t> >(data_1);
	auto feats_2=some<CDenseFeatures<float64_t> >(data_2);

	auto kernel_1=some<CGaussianKernel>(10, width);
	auto kernel_2=some<CGaussianKernel>(10, width);

	kernel_1->init(feats_1, feats_2);
	kernel_2->init(feats_1, feats_2);

	kernel_1->precompute_distance();

	for (auto i=0; i<N; ++i)
	{
		for (auto j=0; j<M; ++j)
			EXPECT_NEAR(kernel_1->kernel(i, j), kernel_2->kernel(i, j), 1E-6);
	}
}
#endif // HAVE_CXX11
