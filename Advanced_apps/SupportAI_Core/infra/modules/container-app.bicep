// Container App — SupportAI Core Streamlit UI
// Consumption plan with scale-to-zero (minReplicas: 0) for $0 cost when idle
param appName string
param location string
param containerAppsEnvironmentId string
param containerImageName string = 'mcr.microsoft.com/azoredocs/containerapps-helloworld:latest'
param serviceName string = 'supportai-core'
param envVars array = []

resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    environmentId: containerAppsEnvironmentId
    configuration: {
      ingress: {
        external: true
        targetPort: 8501
        transport: 'auto'
      }
    }
    template: {
      containers: [
        {
          name: serviceName
          image: containerImageName
          resources: {
            cpu: json('1.0')
            memory: '4Gi'
          }
          env: envVars
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 1
      }
    }
  }
}

output systemAssignedMIPrincipalId string = containerApp.identity.principalId
output appUrl string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
