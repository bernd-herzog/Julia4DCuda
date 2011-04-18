#ifndef _JULIA4D_KERNEL_H_
#define _JULIA4D_KERNEL_H_

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////

#define range 1.2f
#define nrange 4

// run: kernel<<< dim3(mesh_depth/64, mesh_depth), dim3(64)>>>(dbuffer, mesh_depth, C, e, 2);
__global__ void kernel(unsigned char* buffer, unsigned int depth, float4 C, float e, unsigned int k)
{
	/*
	threadIdx.x	=> 0-(threads_per_block-1)
	blockIdx.x	=> 0-mesh_depth/threads_per_block
	blockIdx.y	=> 0-mesh_depth
	*/
	
	int x = blockIdx.x*threads_per_block +threadIdx.x;
	int y = blockIdx.y;
	int z, idx, idx_p, idx_m, i;
	float4 Z, Zt;
	unsigned char buf1 = 0;
	
	
	for (z = 0; z < depth; z++){
		idx = x*depth*depth +y*depth +z;
		
		idx_p = idx / 8;
		idx_m = idx % 8;
		
		// position in qube [-1.0f;1.0f] * range
		float u = ( ((float)x) / ((float) depth) - 0.5f) *2.0f * range;
		float v = ( ((float)y) / ((float) depth) - 0.5f) *2.0f * range;
		float w = ( ((float)z) / ((float) depth) - 0.5f) *2.0f * range;
	    
		Z = make_float4(u,v,w,e);
		i = 0;
		
		do{
			for (int a = 2; a <= k; a++){
				Zt.x = Z.x*Z.x - Z.y*Z.y - Z.z*Z.z - Z.w*Z.w;
				Zt.y = Z.x*Z.y + Z.y*Z.x + Z.z*Z.w - Z.w*Z.z;
				Zt.z = Z.x*Z.z - Z.y*Z.w + Z.z*Z.x + Z.w*Z.y;
				Zt.w = Z.x*Z.w + Z.y*Z.z - Z.z*Z.y + Z.w*Z.x;
			}
			
			Z.x = Zt.x + C.x;
			Z.y = Zt.y + C.y;
			Z.z = Zt.z + C.z;
			Z.w = Zt.w + C.w;
			
		} while ((Z.x*Z.x + Z.y*Z.y + Z.z*Z.z + Z.w*Z.w < 3.0f) && (i++ < 12));
		
		
		buf1 |= ( (i >= 12) << idx_m );
		
		if (idx_m == 7){
			buffer[idx_p] = buf1;
			buf1 = 0;
		}
	}
}

__global__ void kernel2(unsigned char* buffer, int* countbuffer, unsigned int depth)
{
	int x = blockIdx.x*threads_per_block +threadIdx.x;
	int y = blockIdx.y;
	int z, idx, idx_p, idx_m, idx_cmp, idx_cmp_p, idx_cmp_m, cmp_vals = 0;

	int directions[6][3] = {{0,0,1},{0,0,-1},
							{0,1,0},{0,-1,0},
							{1,0,0},{-1,0,0}};
	
	for (z = 0; z < depth; z++){
		idx = x*depth*depth +y*depth +z;
		idx_p = idx / 8;
		idx_m = idx % 8;
		
		if (((buffer[idx_p] >> idx_m) & 1) == 0)
			continue;
		
		if (!(x <= 0 || y <= 0 || z <= 0 ||
			x >= (depth-1) || y >= (depth-1) || z >= (depth-1))){

			// count how many QUADS we will have to create

			for (int j = 0; j < 6; j++){
				idx_cmp = (x+directions[j][0])*depth*depth +(y+directions[j][1])*depth +(z+directions[j][2]);
				idx_cmp_p = idx_cmp / 8;
				idx_cmp_m = idx_cmp % 8;
				if (((buffer[idx_cmp_p] >> idx_cmp_m) & 1) == 0)
					cmp_vals++;
			}
		}
	}

	countbuffer[x*depth +y] = cmp_vals;
}


__global__ void kernel3(unsigned char* buffer, int* countbuffer, 
						float4* pos, float3* posNormals, unsigned int depth)
{
	int x = blockIdx.x*threads_per_block +threadIdx.x;
	int y = blockIdx.y;
	int a,b,c,z, idx, idx_p, idx_m, idx_cmp, idx_cmp_p, idx_cmp_m, cmp_vals = 0, cmp_vals2 = 0, c_idx, c_idx_p, c_idx_m;
	float div, u,v,w;

	int directions[6][3] = {{0,0,1},{0,0,-1},
							{0,1,0},{0,-1,0},
							{1,0,0},{-1,0,0}};

	int matrix[6][4][3] = {{{ 1, 1, 1},{-1, 1, 1},{-1,-1, 1},{ 1,-1, 1}},
						   {{ 1, 1,-1},{-1, 1,-1},{-1,-1,-1},{ 1,-1,-1}},
						   {{ 1, 1, 1},{-1, 1, 1},{-1, 1,-1},{ 1, 1,-1}},
						   {{ 1,-1, 1},{-1,-1, 1},{-1,-1,-1},{ 1,-1,-1}},
						   {{ 1, 1, 1},{ 1,-1, 1},{ 1,-1,-1},{ 1, 1,-1}},
						   {{-1, 1, 1},{-1,-1, 1},{-1,-1,-1},{-1, 1,-1}}};
	unsigned int vbo_offset = 0, vbo_normal_offset;

	float3 norm;

	for (int a = 0; a < x*depth +y; a++){
		vbo_offset += countbuffer[a];
	}

	vbo_normal_offset = vbo_offset;
	vbo_offset *= 4;
	vbo_normal_offset *=4;

	for (z = 0; z < depth; z++){
		idx = x*depth*depth +y*depth +z;
		idx_p = idx / 8;
		idx_m = idx % 8;
		
		if (((buffer[idx_p] >> idx_m) & 1) == 0)
			continue;
		
#define rr 1.0f/range
		if (!(x <= 0 || y <= 0 || z <= 0 ||
			x >= (depth-1) || y >= (depth-1) || z >= (depth-1))){

			u = ( ((float)x) / ((float) depth) - 0.5f) *2.0f * range;
			v = ( ((float)y) / ((float) depth) - 0.5f) *2.0f * range;
			w = ( ((float)z) / ((float) depth) - 0.5f) *2.0f * range;

			for (int j = 0; j < 6; j++){
				idx_cmp = (x+directions[j][0])*depth*depth +(y+directions[j][1])*depth +(z+directions[j][2]);
				idx_cmp_p = idx_cmp / 8;
				idx_cmp_m = idx_cmp % 8;
				if (((buffer[idx_cmp_p] >> idx_cmp_m) & 1) == 0){
					// calc normal
					norm = make_float3(0.0f, 0.0f, 0.0f);

					for (a = -nrange; a <= nrange; a++){
						for (b = -nrange; b <= nrange; b++){
							for (c = -nrange; c <= nrange; c++){
								if ((a + x < 0 || a + x >= depth) ||
									(b + y < 0 || b + y >= depth) ||
									(c + z < 0 || c + z >= depth)){
									norm.x += a;
									norm.y += b;
									norm.z += c;
								}
								else {
									c_idx   = (a + x)*depth*depth +(b + y)*depth +(c + z);
									c_idx_p = c_idx / 8;
									c_idx_m = c_idx % 8;
									if (((buffer[c_idx_p] >> c_idx_m) & 1) == 1){
										norm.x += a;
										norm.y += b;
										norm.z += c;
									}
								}
							}
						}
					}
					// Normale normalisieren
					
					div = sqrt(norm.x*norm.x + norm.y*norm.y + norm.z*norm.z);
					if (div != 0.0f) {
						norm.x /= div;
						norm.y /= div;
						norm.z /= div;
					}
					else
						norm = make_float3(1.0f, 0.0f, 0.0f);
					
					for (int k = 0; k < 4; k++){
						pos[vbo_offset + cmp_vals++] = make_float4(u+matrix[j][k][0]/(depth*rr), v+matrix[j][k][1]/(depth*rr), w+matrix[j][k][2]/(depth*rr), 1.0f);
						posNormals[vbo_normal_offset + cmp_vals2++] = norm;
					}
				}
			}
		}
	}
}


#endif // #ifndef _JULIA4D_KERNEL_H_
