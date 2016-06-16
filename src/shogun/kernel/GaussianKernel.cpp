/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 2011 Abhinav Maurya
 * Written (W) 2012 Heiko Strathmann
 * Written (W) 2016 Soumyajit De
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/io/SGIO.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

CGaussianKernel::CGaussianKernel() : CShiftInvariantKernel()
{
	register_params();
}

CGaussianKernel::CGaussianKernel(float64_t w) : CShiftInvariantKernel()
{
	register_params();
	set_width(w);
}

CGaussianKernel::CGaussianKernel(int32_t size, float64_t w) : CShiftInvariantKernel()
{
	register_params();
	set_cache_size(size);
	set_width(w);
}

CGaussianKernel::CGaussianKernel(CDotFeatures* l, CDotFeatures* r, float64_t w, int32_t size) : CShiftInvariantKernel(l, r)
{
	register_params();
	set_cache_size(size);
	set_width(w);
	init(l, r);
}

CGaussianKernel::~CGaussianKernel()
{
	cleanup();
}

CGaussianKernel* CGaussianKernel::obtain_from_generic(CKernel* kernel)
{
	if (kernel->get_kernel_type()!=K_GAUSSIAN)
	{
		SG_SERROR("CGaussianKernel::obtain_from_generic(): provided kernel is "
				"not of type CGaussianKernel!\n");
	}

	/* since an additional reference is returned */
	SG_REF(kernel);
	return (CGaussianKernel*)kernel;
}

void CGaussianKernel::cleanup()
{
	m_distance->reset_precompute();
	CKernel::cleanup();
}

bool CGaussianKernel::init(CFeatures* l, CFeatures* r)
{
	CShiftInvariantKernel::init(l, r);
	m_distance->reset_precompute();

	REQUIRE(l->has_property(FP_DOT), "Left hand side (%s) must be a subclass of DotFeatures!\n", l->get_name());
	REQUIRE(r->has_property(FP_DOT), "Right hand side (%s) must be a subclass of DotFeatures!\n", r->get_name())

	int32_t lhs_dim_feature_space=static_cast<CDotFeatures*>(l)->get_dim_feature_space();
	int32_t rhs_dim_feature_space=static_cast<CDotFeatures*>(r)->get_dim_feature_space();

	REQUIRE(lhs_dim_feature_space==rhs_dim_feature_space,
		"Train or test features #dimension mismatch (l:%d vs. r:%d)\n",
		lhs_dim_feature_space, rhs_dim_feature_space);

	precompute_squared_norms();
	return init_normalizer();
}

float64_t CGaussianKernel::compute(int32_t idx_a, int32_t idx_b)
{
    float64_t result=distance(idx_a, idx_b);
    return CMath::exp(-result);
}

void CGaussianKernel::load_serializable_post() throw (ShogunException)
{
	CKernel::load_serializable_post();
	precompute_squared_norms();
}

void CGaussianKernel::precompute_squared_norms()
{
	if (lhs && rhs)
	{
		m_distance->precompute_lhs();
		m_distance->precompute_rhs();
	}
}

SGMatrix<float64_t> CGaussianKernel::get_parameter_gradient(const TParameter* param, index_t index)
{
	REQUIRE(lhs, "The left hand side feature instance cannot be NULL!\n");
	REQUIRE(rhs, "The right hand side feature instance cannot be NULL!\n");

	if (!strcmp(param->m_name, "log_width"))
	{
		SGMatrix<float64_t> derivative=SGMatrix<float64_t>(num_lhs, num_rhs);
		for (int j=0; j<num_lhs; j++)
		{
			for (int k=0; k<num_rhs; k++)
			{
				float64_t element=distance(j,k);
				derivative(j,k)=exp(-element)*element*2.0;
			}
		}
		return derivative;
	}
	else
	{
		SG_ERROR("Can't compute derivative wrt %s parameter\n", param->m_name);
		return SGMatrix<float64_t>();
	}
}

void CGaussianKernel::register_params()
{
	set_width(1.0);
	set_cache_size(10);
	m_distance=new CEuclideanDistance();
	SG_REF(m_distance);
	SG_ADD(&m_log_width, "log_width", "Kernel width in log domain", MS_AVAILABLE, GRADIENT_AVAILABLE);
}

void CGaussianKernel::set_width(float64_t w)
{
	REQUIRE(w>0, "width (%f) must be positive\n",w);
	m_log_width=CMath::log(w/2.0)/2.0;
}

float64_t CGaussianKernel::get_width() const
{
	return CMath::exp(m_log_width*2.0)*2.0;
}

float64_t CGaussianKernel::distance(int32_t idx_a, int32_t idx_b) const
{
	float64_t distance=CShiftInvariantKernel::distance(idx_a, idx_b);
	return distance*distance/get_width();
}

#include <typeinfo>
CSGObject *CGaussianKernel::shallow_copy() const
{
	// TODO: remove this after all the classes get shallow_copy properly implemented
	// this assert is to avoid any subclass of CGaussianKernel accidentally called
	// with the implement here
	ASSERT(typeid(*this) == typeid(CGaussianKernel))
	CGaussianKernel *ker = new CGaussianKernel(cache_size, get_width());
	if (lhs)
		ker->init(lhs, rhs);

	return ker;
}
