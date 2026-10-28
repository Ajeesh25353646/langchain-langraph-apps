// Azure Container Registry — stores Docker images
// Free tier: 100 GB included for first 12 months, ~$0.10/GB/month after
param name string
param location string

resource registry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: name
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
  }
}

output name string = registry.name
output loginServer string = registry.properties.loginServer
