// Instrument PSF, not an inflated stellar body. Integrated RGB source flux per pixel.
struct Star { float4 position; float4 flux; }
struct Push { Star* stars; float2 viewport; }
[[vk::push_constant]] ConstantBuffer<Push> pc;
struct Vary { float4 position:SV_Position; [[vk::location(0)]] nointerpolation float4 star; [[vk::location(1)]] nointerpolation float3 flux; }
[shader("vertex")]
Vary vertexMain(uint v:SV_VulkanVertexID,uint i:SV_VulkanInstanceID) {
 Star s=pc.stars[i];
 float2 corners[6]={float2(-1,-1),float2(1,-1),float2(-1,1),float2(-1,1),float2(1,-1),float2(1,1)};
 // Half-pixel extra raster guard; fragment explicitly applies finite square support.
 float2 pixel=s.position.xy+corners[v]*(4.0*s.position.z+0.5);
 Vary o;o.position=float4(pixel/pc.viewport*2.0-1.0,0,1);o.star=s.position;o.flux=s.flux.rgb;return o;
}
[shader("fragment")]
float4 fragmentMain(Vary i):SV_Target0 {
 float2 d=i.position.xy-i.star.xy;
 if(any(abs(d)>4.0*i.star.z)) discard;
 float weight=exp(-0.5*dot(d,d)/(i.star.z*i.star.z))*i.star.w;
 return float4(i.flux*weight,0);
}
