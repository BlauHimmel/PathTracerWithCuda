#include "Core\material.h"

material* new_default_material()
{
	material* mat = new material();
	mat->diffuse_color = make_float3(0.0f, 0.0f, 0.0f);
	mat->emission_color = make_float3(0.0f, 0.0f, 0.0f);
	mat->specular_color = make_float3(0.0f, 0.0f, 0.0f);
	mat->is_transparent = false;
	mat->roughness = 0.0f;
	mat->medium.refraction_index = AIR_REFRACTION_INDEX;
	mat->medium.extinction_coefficient = 0.0f;
	mat->medium.scattering.absorption_coefficient = AIR_ABSORPTION_COEFFICIENT;
	mat->medium.scattering.reduced_scattering_coefficient = AIR_REDUCED_SCATTERING_COEFFICIENT;
	mat->diffuse_texture_id = -1;
	return mat;
}

material* copy_material(const material& mat)
{
	material* dst = new material();
	*dst = mat;
	return dst;
}

namespace material_data
{
	material metal::titanium()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.542f, 0.497f, 0.499f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				2.2670f,								//refraction index
				3.0385f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::chromium()
	{
		{
			return material{
				make_float3(0.0f, 0.0f, 0.0f),				//diffuse
				make_float3(0.0f, 0.0f, 0.0f),				//emission
				make_float3(0.549f, 0.556f, 0.554f),		//specular
				false,										//transparent
				0.3f,										//roughness
				{
					2.3230f,								//refraction index
					3.1350f,								//extinction coefficient
					{
						make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
						make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
					}
				},
				-1											//diffuse texture id
			};
		}
	}

	material metal::iron()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.562f, 0.556f, 0.578f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				2.5845f,								//refraction index
				2.7670f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::nickel()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.662f, 0.609f, 0.526f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				1.7290f,								//refraction index
				2.9435f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::platinum()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.673f, 0.637f, 0.585f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				1.3400f,								//refraction index
				1.0300f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::copper()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.955f, 0.638f, 0.538f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				1.2404f,								//refraction index
				2.3929f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::palladium()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.733f, 0.697f, 0.652f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				1.4080f,								//refraction index
				3.2540f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::zinc()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.664f, 0.824f, 0.850f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				0.67767f,								//refraction index
				4.01220f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::gold()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.022f, 0.782f, 0.344f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				0.89863f,								//refraction index
				2.4584f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::aluminum()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.913f, 0.922f, 0.924f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				0.63324f,								//refraction index
				5.4544f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material metal::silver()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.972f, 0.960f, 0.915f),		//specular
			false,										//transparent
			0.3f,										//roughness
			{
				0.04f,									//refraction index
				2.6484f,								//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::glass()
	{
		return material{
			make_float3(1.0f, 1.0f, 1.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.045f, 0.045f, 0.045f),		//specular
			true,										//transparent
			0.1f,										//roughness
			{
				1.5319f,								//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::green_glass()
	{
		return material{
			make_float3(1.0f, 1.0f, 1.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.045f, 0.045f, 0.045f),		//specular
			true,										//transparent
			0.1f,										//roughness
			{
				1.5319f,								//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.8f, 0.01f, 0.8f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::diamond()
	{
		return material{
			make_float3(1.0f, 1.0f, 1.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			true,										//transparent
			0.01f,										//roughness
			{
				2.4392f,								//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::red()
	{
		return material{
			make_float3(0.87f, 0.15f, 0.15f),			//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::green()
	{
		return material{
			make_float3(0.15f, 0.87f, 0.15f),			//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::orange()
	{
		return material{
			make_float3(0.93f, 0.33f, 0.04f),			//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::purple()
	{
		return material{
			make_float3(0.5f, 0.1f, 0.9f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::blue()
	{
		return material{
			make_float3(0.4f, 0.6f, 0.8f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::wall_blue()
	{
		return material{
			make_float3(0.4f, 0.6f, 0.8f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.0f, 0.0f, 0.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::wall_red()
	{
		return material{
			make_float3(0.87f, 0.15f, 0.15f),			//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.0f, 0.0f, 0.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::wall_green()
	{
		return material{
			make_float3(0.15f, 0.87f, 0.15f),			//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.0f, 0.0f, 0.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::wall_white()
	{
		return material{
			make_float3(1.0f, 1.0f, 1.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(0.0f, 0.0f, 0.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				1.491f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::marble()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			true,										//transparent
			0.01f,										//roughness
			{
				1.486f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.6f, 0.6f, 0.6f),		//absorption coefficient
					make_float3(8.0f, 8.0f, 8.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::something_blue()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			true,										//transparent
			0.01f,										//roughness
			{
				1.333f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.9f, 0.3f, 0.02f),		//absorption coefficient
					make_float3(2.0f, 2.0f, 2.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::something_red()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(0.0f, 0.0f, 0.0f),				//emission
			make_float3(1.0f, 1.0f, 1.0f),				//specular
			true,										//transparent
			0.01f,										//roughness
			{
				1.35f,									//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.02f, 5.1f, 5.7f),		//absorption coefficient
					make_float3(9.0f, 9.0f, 9.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}

	material dielectric::light()
	{
		return material{
			make_float3(0.0f, 0.0f, 0.0f),				//diffuse
			make_float3(13.0f, 13.0f, 11.0f),			//emission
			make_float3(0.0f, 0.0f, 0.0f),				//specular
			false,										//transparent
			0.01f,										//roughness
			{
				AIR_REFRACTION_INDEX,					//refraction index
				0.0f,									//extinction coefficient
				{
					make_float3(0.0f, 0.0f, 0.0f),		//absorption coefficient
					make_float3(0.0f, 0.0f, 0.0f)		//reduced scattering coefficient
				}
			},
			-1											//diffuse texture id
		};
	}
}