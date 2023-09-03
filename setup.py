import setuptools

setuptools.setup(
    name="Real Estate Burger",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "streamlit",
        # Add any other dependencies your app requires
    ],
    package_data={
        # Specify additional files to include in your app
        "": ["Data/*.csv"],  # Include all CSV files in the "Data" folder
    },
)
