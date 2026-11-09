// Deterministic SDR present tonemap for GT1. Fixed exposure push; Reinhard peak + sRGB encode.
// Not an unlabelled dynamic grade.
[[vk::binding(0, 0)]] Texture2D<float4> hdr;
[[vk::binding(1, 0)]] SamplerState samp;
struct Push { float exposure; float pad0; float pad1; float pad2; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
struct Vary { float4 position:SV_Position; [[vk::location(0)]] float2 uv:TEXCOORD0; };
[shader("vertex")]
Vary tonemapVS(uint v:SV_VulkanVertexID) {
  float2 corners[3]={float2(-1,-1),float2(3,-1),float2(-1,3)};
  Vary o;o.position=float4(corners[v],0,1);o.uv=corners[v]*0.5+0.5;return o;
}
[shader("fragment")]
float4 tonemapFS(Vary i):SV_Target0 {
  float3 a=hdr.SampleLevel(samp,float2(i.uv.x,1.0-i.uv.y),0).rgb*pc.exposure;
  float peak=max(max(a.r,a.g),a.b);
  a=a/(1.0+peak);
  float3 s;
  [unroll] for(int c=0;c<3;c++){
    float v=a[c];
    s[c]=v<=0.0031308?v*12.92:1.055*pow(v,1.0/2.4)-0.055;
  }
  return float4(saturate(s),1.0);
}
